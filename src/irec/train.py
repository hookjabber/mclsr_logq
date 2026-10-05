"""Exploratory training entry point: ``python -m irec.train --params <config>`` (validation and test
callbacks as configured, TensorBoard logs, best and final checkpoints). The pre-registered protocol with a
single test read is ``scripts/train_confirmatory.py``; both share ``irec.training``.
"""

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

import json
import os
import torch

from irec.training import (  # noqa: F401  (re-exported for the scripts)
    atomic_save,
    best_checkpoint_path,
    seed_val,
    train,
    unwrap_state_dict,
)

logger = create_logger(name=__name__)


def main():
    config = parse_args()
    fix_random_seed(
        config.get('seed', seed_val),
        deterministic=config.get('deterministic', False),
    )

    if config.get('use_wandb', False):
        import wandb  # optional dependency: required only when use_wandb is on

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
    logger.info(f'Experiment config saved to: {config_save_path}')

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
        # accept both bare state dicts and checkpoint wrappers; NB this
        # restores WEIGHTS only (optimizer/step/RNG resume is not implemented —
        # this is a warm start, not a resume)
        model.load_state_dict(unwrap_state_dict(checkpoint))

    loss_function = BaseLoss.create_from_config(config['loss'])
    if isinstance(loss_function, torch.nn.Module):
        loss_function = loss_function.to(DEVICE)

    optimizer = BaseOptimizer.create_from_config(
        config['optimizer'],
        model=model,
        extra_parameters=(list(loss_function.parameters()) if isinstance(loss_function, torch.nn.Module) else []),
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

    # the effective seed always lands in the name so that multi-seed runs of
    # one config (and default-seed reruns) never overwrite each other
    checkpoint_stem = '{}_seed{}'.format(
        config['experiment_name'],
        config.get('seed', seed_val),
    )
    primary_metric = config.get('best_metric')
    if isinstance(primary_metric, list):
        primary_metric = primary_metric[0] if primary_metric else None
    ensure_checkpoints_dir()

    def save_best(metric_name, best):
        path = best_checkpoint_path(checkpoint_stem, metric_name, primary_metric)
        atomic_save(best['state'], path)
        logger.debug(
            'Saved best checkpoint (by {} = {:.5f} @ step {}) as {}'.format(
                metric_name,
                best['value'],
                best['step'],
                path,
            ),
        )

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
        epochs_threshold=config.get('epochs_threshold', 40),
        on_improvement=save_best,  # best states hit the disk as they appear
    )

    logger.debug('Saving model...')
    checkpoint_path = './checkpoints/{}_final_state.pth'.format(
        checkpoint_stem,
    )
    atomic_save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))
    for metric_name, best in best_checkpoints.items():
        save_best(metric_name, best)  # idempotent: same path, same state

    if config.get('use_wandb', False):
        import wandb

        wandb.finish()


if __name__ == '__main__':
    main()
