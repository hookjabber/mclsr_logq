#!/usr/bin/env python
"""Confirmatory (pre-registered) training runner: test is opened ONCE.

Differences from the exploratory `train` entrypoint:
- the eval (test) callback is STRIPPED from the config before training — no
  test METRIC is ever computed while the model trains (the split file is read
  at dataset construction, but its dataloader is only built after training);
- a hard step limit is mandatory (config `train_steps_num` or --max-steps);
- graphs must be train-only (use_train_data_only must not be disabled);
- one best checkpoint is tracked per each --select validation metric;
- protocol: ONE test evaluation per pre-registered validation-selected
  checkpoint (with two --select metrics the test split is scored twice — both
  results are reported, none is discarded);
- the mandatory JSON report records validation value/step/epoch behind every
  selection, sha256 of the config / count-table artifacts / checkpoints, git
  commit + dirty status and the torch version.

Usage (one seed per invocation; loop seeds from the shell):
    python scripts/train_confirmatory.py --params configs/train/toys/03_graph.json \
        --seed 1 --select validation/ndcg@20 validation/recall@1000 \
        --output results/toys_03_seed1.json
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint import evaluate, get_inference_config  # noqa: E402

import irec.utils  # noqa: E402
from irec.callbacks import BaseCallback  # noqa: E402
from irec.dataloader import BaseDataloader  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.loss import BaseLoss  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.optimizer import BaseOptimizer  # noqa: E402
from irec.train import train  # noqa: E402
from irec.utils import DEVICE, ensure_checkpoints_dir, fix_random_seed  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True,
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='hard step limit (falls back to config train_steps_num)')
    parser.add_argument(
        '--select', nargs='+',
        default=['validation/ndcg@20', 'validation/recall@1000'],
        help='validation metrics; one best checkpoint is kept per metric',
    )
    parser.add_argument('--output', required=True, help='json report path')
    args = parser.parse_args()

    with open(args.params) as f:
        config = json.load(f)

    dataset_config = config.get('dataset', {})
    if dataset_config.get('use_train_data_only') is False:
        raise SystemExit(
            'confirmatory runs must keep use_train_data_only=true '
            '(graphs built with val/test interactions leak the split)'
        )

    step_limit = args.max_steps or config.get('train_steps_num')
    if not step_limit:
        raise SystemExit(
            'confirmatory runs need a hard step limit: set train_steps_num '
            'in the config or pass --max-steps'
        )

    # fail before training if a pre-registered --select metric is not even
    # computed by the validation callback (train() would drop it silently)
    def collect_validation_metrics(node, names):
        if isinstance(node, dict):
            if node.get('type') == 'validation':
                for metric_name in node.get('metrics', {}):
                    names.add('validation/{}'.format(metric_name))
            for value in node.values():
                collect_validation_metrics(value, names)
        elif isinstance(node, list):
            for item in node:
                collect_validation_metrics(item, names)
    available_metrics = set()
    collect_validation_metrics(config.get('callback', {}), available_metrics)
    missing = sorted(set(args.select) - available_metrics)
    if available_metrics and missing:
        raise SystemExit(
            'pre-registered --select metrics are not computed by the '
            'validation callback: {} (available: {})'.format(
                missing, sorted(available_metrics),
            )
        )

    config['seed'] = args.seed
    fix_random_seed(args.seed, deterministic=config.get('deterministic', False))

    experiment = '{}_confirm_seed{}'.format(config['experiment_name'], args.seed)
    writer = irec.utils.tensorboards.TensorboardWriter(experiment)
    irec.utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = writer
    with open(os.path.join(writer.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    dataset = BaseDataset.create_from_config(config['dataset'])
    train_sampler, validation_sampler, test_sampler = dataset.get_samplers()
    train_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['train'], dataset=train_sampler, **dataset.meta,
    )
    validation_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=validation_sampler,
        **dataset.meta,
    )

    # the test callback never runs during confirmatory training
    train_callback_config = copy.deepcopy(config['callback'])
    kept = [
        cb for cb in train_callback_config['callbacks'] if cb['type'] != 'eval'
    ]
    if len(kept) == len(train_callback_config['callbacks']):
        raise SystemExit('no eval callback found in the config to strip')
    train_callback_config['callbacks'] = kept

    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    loss_function = BaseLoss.create_from_config(config['loss'])
    optimizer = BaseOptimizer.create_from_config(config['optimizer'], model=model)
    callback = BaseCallback.create_from_config(
        train_callback_config,
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        eval_dataloader=validation_dataloader,  # eval slot never used, keep test out
        optimizer=optimizer,
        **dataset.meta,
    )

    best_checkpoints = train(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config.get('train_epochs_num'),
        step_cnt=step_limit,
        best_metric=args.select,
    )
    if not best_checkpoints:
        raise SystemExit('no best checkpoint captured — check --select names')
    missing_after_train = sorted(set(args.select) - set(best_checkpoints))
    if missing_after_train:
        raise SystemExit(
            'pre-registered --select metrics missing from the validation '
            'stream: {} (captured: {})'.format(
                missing_after_train, sorted(best_checkpoints),
            )
        )

    # the test dataloader exists only from this point on
    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'], dataset=test_sampler, **dataset.meta,
    )

    ensure_checkpoints_dir()
    inference_config = get_inference_config(config, 'eval')
    artifact_hashes = {}
    def collect_paths(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.startswith('path_to') and os.path.isfile(value):
                    artifact_hashes[value] = file_sha256(value)
                else:
                    collect_paths(value)
        elif isinstance(node, list):
            for item in node:
                collect_paths(item)
    collect_paths(config)

    try:
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True,
        ).strip())
    except Exception:
        dirty = None

    report = {
        'experiment': experiment,
        'params': args.params,
        'params_sha256': file_sha256(args.params),
        'artifact_sha256': artifact_hashes,
        'seed': args.seed,
        'step_limit': step_limit,
        'git_commit': git_commit(),
        'git_dirty': dirty,
        'torch_version': torch.__version__,
        'protocol': 'one test evaluation per pre-registered '
                    'validation-selected checkpoint; shared no-progress '
                    'patience across selection metrics',
        'results': {},
    }
    for metric_name, best in best_checkpoints.items():
        slug = metric_name.replace('/', '_').replace('@', '_at_')
        path = './checkpoints/{}_best_{}.pth'.format(experiment, slug)
        torch.save(best['state'], path)
        model.load_state_dict(best['state'])
        test_metrics = evaluate(
            model=model,
            dataloader=eval_dataloader,
            metric_configs=inference_config['metrics'],
            pred_prefix=inference_config['pred_prefix'],
            labels_prefix=inference_config['labels_prefix'],
            meta=dataset.meta,
        )
        report['results'][metric_name] = {
            'selection_metric': metric_name,
            'validation_value': best['value'],
            'step': best['step'],
            'epoch': best['epoch'],
            'checkpoint': path,
            'checkpoint_sha256': file_sha256(path),
            'test': test_metrics,
        }
        print('[{}] -> {}'.format(metric_name, {
            k: round(v, 5) for k, v in test_metrics.items()
            if k in ('ndcg@20', 'recall@1000')
        }))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print('report ->', args.output)


if __name__ == '__main__':
    main()
