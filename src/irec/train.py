import irec.utils
from irec.utils import (
    parse_args,
    create_logger,
    DEVICE,
    fix_random_seed,
    ensure_checkpoints_dir,
)

from irec.callbacks import BaseCallback
from irec.dataset import BaseDataset
from irec.dataloader import BaseDataloader
from irec.loss import BaseLoss
from irec.models import BaseModel
from irec.optimizer import BaseOptimizer

import copy
import json
import os
import torch
import wandb

logger = create_logger(name=__name__)
seed_val = 42


def unwrap_state_dict(checkpoint):
    """Accept bare state dicts and CheckpointCallback/eval wrappers alike."""
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def train(
    dataloader,
    model,
    optimizer,
    loss_function,
    callback,
    epoch_cnt=None,
    step_cnt=None,
    best_metric=None,
):
    """`best_metric`: None (final state only), one metric name, or a LIST of
    names — one independently tracked best checkpoint per metric. Returns
    {metric_name: {'state': cpu_state_dict, 'value', 'step', 'epoch'}}
    (empty when best_metric is None). States are copied to CPU immediately so
    several tracked bests do not accumulate on the GPU. NB the no-progress
    patience is shared: an improvement in ANY tracked metric resets it."""
    step_num = 0
    epoch_num = 0

    epochs_threshold = 40

    best_metrics = (
        [best_metric] if isinstance(best_metric, str)
        else list(best_metric or [])
    )
    current_values = {name: None for name in best_metrics}
    best_checkpoints = {}
    best_epoch = 0

    logger.debug('Start training...')

    while (epoch_cnt is None or epoch_num < epoch_cnt) and (
        step_cnt is None or step_num < step_cnt
    ):
        if best_epoch + epochs_threshold < epoch_num:
            logger.debug(
                'There is no progress during {} epochs. Finish training'.format(
                    epochs_threshold,
                ),
            )
            break

        logger.debug(f'Start epoch {epoch_num}')
        for step, batch in enumerate(dataloader):
            if step_cnt is not None and step_num >= step_cnt:
                break
            batch_ = copy.deepcopy(batch)

            model.train()

            for key, values in batch_.items():
                batch_[key] = batch_[key].to(DEVICE)

            batch_.update(model(batch_))
            loss = loss_function(batch_)

            optimizer.step(loss)
            callback(batch_, step_num)
            step_num += 1

            for name in best_metrics:
                if name not in batch_:
                    continue
                value = batch_[name]
                # strict improvement: on an exact tie keep the FIRST best, the
                # same convention summarize_runs.py uses to pick the step
                if current_values[name] is None or current_values[name] < value:
                    current_values[name] = value
                    state = {
                        key: (
                            tensor.detach().cpu().clone()
                            if torch.is_tensor(tensor) else copy.deepcopy(tensor)
                        )
                        for key, tensor in model.state_dict().items()
                    }
                    best_checkpoints[name] = {
                        'state': state,
                        'value': float(value),
                        'step': step_num - 1,
                        'epoch': epoch_num,
                    }
                    best_epoch = epoch_num

        epoch_num += 1
    logger.debug('Training procedure has been finished!')
    return best_checkpoints


def main():
    config = parse_args()
    fix_random_seed(
        config.get('seed', seed_val),
        deterministic=config.get('deterministic', False),
    )

    if config.get('use_wandb', False):
        wandb.init(
            project='irec',
            name=config['experiment_name'],
            sync_tensorboard=True,
        )

    tensorboard_writer = irec.utils.tensorboards.TensorboardWriter(config['experiment_name'])
    irec.utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = tensorboard_writer

    log_dir = tensorboard_writer.log_dir
    config_save_path = os.path.join(log_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))
    logger.debug('Current DEVICE: {}'.format(DEVICE))
    logger.info(f"Experiment config saved to: {config_save_path}")


    dataset = BaseDataset.create_from_config(config['dataset'])

    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['train'],
        dataset=train_sampler,
        **dataset.meta,
    )

    validation_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=validation_sampler,
        **dataset.meta,
    )

    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=test_sampler,
        **dataset.meta,
    )

    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(
        DEVICE,
    )
    if 'checkpoint' in config:
        ensure_checkpoints_dir()
        checkpoint_path = os.path.join(
            './checkpoints',
            f'{config["checkpoint"]}.pth',
        )
        logger.debug('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        logger.debug(checkpoint.keys())
        # accept both bare state dicts and CheckpointCallback wrappers; NB this
        # restores WEIGHTS only (optimizer/step/RNG resume is not implemented —
        # this is a warm start, not a resume)
        model.load_state_dict(unwrap_state_dict(checkpoint))

    loss_function = BaseLoss.create_from_config(config['loss'])

    optimizer = BaseOptimizer.create_from_config(
        config['optimizer'],
        model=model,
    )

    callback = BaseCallback.create_from_config(
        config['callback'],
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        **dataset.meta,
    )

    # TODO add verbose option for all callbacks, multiple optimizer options (???)
    # TODO create pre/post callbacks
    logger.debug('Everything is ready for training process!')

    # Train process
    best_checkpoints = train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=config.get('train_steps_num'),
        best_metric=config.get('best_metric'),
    )

    logger.debug('Saving model...')
    ensure_checkpoints_dir()
    # the effective seed always lands in the name so that multi-seed runs of
    # one config (and default-seed reruns) never overwrite each other
    checkpoint_stem = '{}_seed{}'.format(
        config['experiment_name'],
        config.get('seed', seed_val),
    )
    checkpoint_path = './checkpoints/{}_final_state.pth'.format(
        checkpoint_stem,
    )
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))

    primary_metric = config.get('best_metric')
    if isinstance(primary_metric, list):
        primary_metric = primary_metric[0] if primary_metric else None
    for metric_name, best in best_checkpoints.items():
        if metric_name == primary_metric:
            suffix = 'best_state'  # historical name for the primary metric
        else:
            suffix = 'best_{}'.format(
                metric_name.replace('/', '_').replace('@', '_at_'),
            )
        best_checkpoint_path = './checkpoints/{}_{}.pth'.format(
            checkpoint_stem, suffix,
        )
        torch.save(best['state'], best_checkpoint_path)
        logger.debug(
            'Saved best checkpoint (by {} = {:.5f} @ step {}) as {}'.format(
                metric_name,
                best['value'],
                best['step'],
                best_checkpoint_path,
            ),
        )

    if config.get('use_wandb', False):
        wandb.finish()


if __name__ == '__main__':
    main()
