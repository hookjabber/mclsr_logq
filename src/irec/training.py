"""The training loop and checkpoint helpers shared by the exploratory and the confirmatory runners."""

from irec.utils import (
    create_logger,
    DEVICE,
)


import copy
import os
import torch

from irec.dataset.negative_samplers.popular import PopularNegativeSampler
from irec.loss.retrieval import SampledLogQLoss

logger = create_logger(name=__name__)
seed_val = 42


def validate_sampling_proposals(train_sampler, loss_function):
    sampler = getattr(train_sampler, '_negative_sampler', None)
    if not isinstance(sampler, PopularNegativeSampler) or not isinstance(loss_function, torch.nn.Module):
        return
    for component in loss_function.modules():
        if not isinstance(component, SampledLogQLoss) or component._logq_lambda == 0:
            continue
        item_ids = torch.as_tensor(sampler._item_ids, dtype=torch.long)
        log_q = component._log_q_table.detach().cpu().double()
        if item_ids.numel() == 0 or item_ids.max() >= log_q.numel():
            raise ValueError('Explicit-negative count table does not cover the sampler base proposal')
        expected = torch.as_tensor(sampler._probs, dtype=torch.float64)
        actual = torch.softmax(log_q[item_ids], dim=0)
        if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-12):
            raise ValueError(
                'Explicit-negative count table does not match the sampler base proposal; '
                'popular sampling requires event frequencies over the loaded training rows, '
                'including targets and repeated occurrences, not context-inclusion counts'
            )


def unwrap_state_dict(checkpoint):
    """Accept bare state dicts and eval wrappers alike."""
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def best_checkpoint_path(checkpoint_stem, metric_name, primary_metric):
    if metric_name == primary_metric:
        suffix = 'best_state'  # historical name for the primary metric
    else:
        suffix = 'best_{}'.format(
            metric_name.replace('/', '_').replace('@', '_at_'),
        )
    return './checkpoints/{}_{}.pth'.format(checkpoint_stem, suffix)


def atomic_save(obj, path):
    """Write to a temp file next to `path`, then rename: a run killed in the
    middle of a save never leaves a truncated checkpoint behind."""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def train(
    dataloader,
    model,
    optimizer,
    loss_function,
    callback,
    epoch_cnt=None,
    step_cnt=None,
    best_metric=None,
    epochs_threshold=40,
    on_improvement=None,
):
    """`best_metric`: None (final state only), one metric name, or a LIST of
    names — one independently tracked best checkpoint per metric. Returns
    {metric_name: {'state': cpu_state_dict, 'value', 'step', 'epoch'}}
    (empty when best_metric is None). States are copied to CPU immediately so
    several tracked bests do not accumulate on the GPU. NB the no-progress
    patience is shared: an improvement in ANY tracked metric resets it.
    `on_improvement(metric_name, best)` (optional) is called right after a
    tracked best is updated — used to persist the best state immediately, so
    an interrupted run (Ctrl+C, node reboot) still leaves its best checkpoint
    on disk instead of only in memory."""
    step_num = 0
    epoch_num = 0

    best_metrics = [best_metric] if isinstance(best_metric, str) else list(best_metric or [])
    current_values = {name: None for name in best_metrics}
    best_checkpoints = {}
    best_epoch = 0

    logger.debug('Start training...')

    while (epoch_cnt is None or epoch_num < epoch_cnt) and (step_cnt is None or step_num < step_cnt):
        if best_metrics and best_epoch + epochs_threshold < epoch_num:
            logger.debug(
                'There is no progress during {} epochs. Finish training'.format(
                    epochs_threshold,
                ),
            )
            break

        logger.debug(f'Start epoch {epoch_num}')
        epoch_start_step = step_num
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
                        key: (tensor.detach().cpu().clone() if torch.is_tensor(tensor) else copy.deepcopy(tensor))
                        for key, tensor in model.state_dict().items()
                    }
                    best_checkpoints[name] = {
                        'state': state,
                        'value': float(value),
                        'step': step_num - 1,
                        'epoch': epoch_num,
                    }
                    best_epoch = epoch_num
                    if on_improvement is not None:
                        on_improvement(name, best_checkpoints[name])

        if step_num == epoch_start_step:
            raise ValueError('Training dataloader produced no batches')
        epoch_num += 1
    logger.debug('Training procedure has been finished!')
    return best_checkpoints
