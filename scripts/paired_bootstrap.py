#!/usr/bin/env python
"""Paired, per-user uncertainty for the headline comparisons.

For every ``arm:baseline`` pair and every seed, the validation-selected checkpoints of both arms
are evaluated on the same test users; the per-user difference of a metric is kept. The pooled
mean difference over (user, seed) gets a 95% bootstrap interval (resampling users, all seeds of a
user together, 10 000 draws) and a two-sided p-value from a sign-flip permutation test (flipping
the sign of all of a user's differences together, 10 000 draws). Per-seed means are reported too.

    python scripts/paired_bootstrap.py --prefix beauty --config-dir configs/train/beauty \
        --seeds 1 2 3 --pairs 02_logq_downstream:01_orig 03_graph:03_graph_l00 \
        --output results/stats/beauty_paired_bootstrap.md
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoint import get_inference_config  # noqa: E402

from irec.dataloader import BaseDataloader  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.metric import BaseMetric  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.train import unwrap_state_dict  # noqa: E402
from irec.utils import DEVICE  # noqa: E402

METRICS = {'ndcg@20': {'type': 'mclsr-ndcg', 'k': 20}, 'recall@1000': {'type': 'mclsr-recall', 'k': 1000}}
_DATASETS = {}


def dataset_for(config):
    key = json.dumps(config['dataset'], sort_keys=True)
    if key not in _DATASETS:
        dataset = BaseDataset.create_from_config(config['dataset'])
        _, _, test_sampler = dataset.get_samplers()
        loader = BaseDataloader.create_from_config(
            config['dataloader']['validation'], dataset=test_sampler, **dataset.meta
        )
        _DATASETS[key] = (dataset, loader)
    return _DATASETS[key]


def per_user_metrics(config, checkpoint):
    dataset, loader = dataset_for(config)
    inference = get_inference_config(config, 'eval')
    pred_prefix, labels_prefix = inference['pred_prefix'], inference['labels_prefix']
    metrics = {name: BaseMetric.create_from_config(cfg, **dataset.meta) for name, cfg in METRICS.items()}
    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    model.load_state_dict(unwrap_state_dict(torch.load(checkpoint, map_location=DEVICE)))
    model.eval()
    values = {name: [] for name in metrics}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch[pred_prefix] = model(batch)
            batch = {k: v.cpu() for k, v in batch.items()}
            for name, metric in metrics.items():
                values[name].extend(metric(inputs=batch, pred_prefix=pred_prefix, labels_prefix=labels_prefix))
    return {name: np.asarray(v, dtype=np.float64) for name, v in values.items()}


def load_arm(prefix, arm, seed, config_dir, confirm_dir, cache):
    key = (arm, seed)
    if key not in cache:
        report = json.load(open(os.path.join(confirm_dir, f'{prefix}_{arm}_seed{seed}.json')))
        config = json.load(open(os.path.join(config_dir, f'{arm}.json')))
        checkpoint = report['results']['validation/ndcg@20']['checkpoint']
        cache[key] = per_user_metrics(copy.deepcopy(config), checkpoint)
    return cache[key]


def paired_stats(D, rng, draws=10000):
    """D: (users, seeds) matrix of per-user differences."""
    n = D.shape[0]
    row_means = D.mean(axis=1)  # one number per user (mean over seeds)
    observed = row_means.mean()
    idx = rng.integers(0, n, size=(draws, n))
    boot = row_means[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(draws, n))
    perm = (signs * row_means[None, :]).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (draws + 1)
    return observed, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--config-dir', required=True)
    ap.add_argument('--confirm-dir', default='results/confirm')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--pairs', nargs='+', required=True, help='arm:baseline')
    ap.add_argument('--output', required=True)
    ap.add_argument('--draws', type=int, default=10000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cache = {}
    lines = [
        '| arm − baseline | metric | per-seed Δ | mean Δ | 95% CI (user bootstrap) | p (sign-flip) | users × seeds |',
        '|---|---|---|---|---|---|---|',
    ]
    out = {}
    for pair in args.pairs:
        arm, base = pair.split(':')
        per_metric = {name: [] for name in METRICS}
        for seed in args.seeds:
            a = load_arm(args.prefix, arm, seed, args.config_dir, args.confirm_dir, cache)
            b = load_arm(args.prefix, base, seed, args.config_dir, args.confirm_dir, cache)
            for name in METRICS:
                assert a[name].shape == b[name].shape, 'arm and baseline must score the same users'
                per_metric[name].append(a[name] - b[name])
        for name in METRICS:
            D = np.stack(per_metric[name], axis=1)  # (users, seeds)
            obs, lo, hi, p = paired_stats(D, rng, args.draws)
            seed_means = ', '.join(f'{m:+.4f}' for m in D.mean(axis=0))
            lines.append(
                f'| {arm} − {base} | {name} | {seed_means} | {obs:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {D.shape[0]} × {D.shape[1]} |'
            )
            out[f'{arm}:{base}:{name}'] = {
                'per_seed': D.mean(axis=0).tolist(),
                'mean': obs,
                'ci95': [lo, hi],
                'p_sign_flip': p,
                'users': int(D.shape[0]),
                'seeds': args.seeds,
            }
            print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f'# Paired per-user comparisons — {args.prefix}\n\n')
        f.write(
            'Validation-selected checkpoints (ndcg@20) of both arms scored on the same test users; '
            'differences pooled over seeds; CI by user bootstrap (10 000 draws, all seeds of a user resampled together); '
            'p by sign-flip permutation.\n\n'
        )
        f.write('\n'.join(lines) + '\n')
    with open(os.path.splitext(args.output)[0] + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('written', args.output)


if __name__ == '__main__':
    main()
