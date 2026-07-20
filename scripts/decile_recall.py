#!/usr/bin/env python
"""Popularity-stratified recall@k from a saved checkpoint.

Splits the item catalog into equal-size popularity bins by TRAIN frequency
(item_counts.pkl, the same table logQ uses), then reports per-bin micro recall:
    recall_bin = (# hit target events in bin) / (# target events in bin)
with a cluster bootstrap over users for confidence intervals. The overall macro
recall (mean of per-user recalls, the tensorboard convention) is printed as a
cross-check against the training curves.

Usage:
    python scripts/decile_recall.py --params configs/train/grid/09_item_only.json \
        --checkpoint checkpoints/mclsr_grid_09_item_only_Clothing_best_state.pth \
        --output decile_09.json
"""
import argparse
import json
import pickle

import numpy as np
import torch

from irec.dataloader import BaseDataloader
from irec.dataset import BaseDataset
from irec.models import BaseModel
from irec.utils import DEVICE, fix_random_seed


def collect_per_user(model, dataloader, k, item_bins, num_bins):
    """per-user (hits, totals) per bin + per-user plain recall@k"""
    hits, totals, user_recalls = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            for key, value in batch.items():
                batch[key] = value.to(DEVICE)
            preds = model(batch)[:, :k].cpu()  # (B, k) item ids
            labels_flat = batch['labels.ids'].cpu()
            lengths = batch['labels.length'].cpu()
            offset = 0
            for i in range(preds.shape[0]):
                n = int(lengths[i].item())
                targets = labels_flat[offset:offset + n]
                offset += n
                if n == 0:
                    continue
                hit_mask = torch.isin(targets, preds[i]).numpy()
                bins = item_bins[targets.numpy()]
                hits.append(np.bincount(bins, weights=hit_mask, minlength=num_bins))
                totals.append(np.bincount(bins, minlength=num_bins))
                user_recalls.append(hit_mask.mean())
    return np.array(hits), np.array(totals), np.array(user_recalls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--counts', default='./data/Clothing/item_counts.pkl',
                        help='TRAIN frequency table used to build the bins')
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--num-bins', type=int, default=10)
    parser.add_argument('--tie-preserving', action='store_true',
                        help='never split items of equal train count across '
                        'bins (bins become unequal-sized but honestly ordered)')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--split', choices=['validation', 'eval'], default='eval')
    parser.add_argument('--baseline-checkpoint', default=None,
                        help='second checkpoint (same config family): report '
                        'per-bin PAIRED differences main - baseline with a '
                        'cluster bootstrap over the shared users')
    parser.add_argument('--baseline-params', default=None,
                        help='config for the baseline checkpoint (defaults to --params)')
    parser.add_argument('--output', default=None, help='write the table as json')
    args = parser.parse_args()

    fix_random_seed(42)
    with open(args.params) as f:
        config = json.load(f)
    with open(args.counts, 'rb') as f:
        counts = np.asarray(pickle.load(f))

    # catalog bins over real items (1..num_items), ranked by train count;
    # bin 0 = least popular. Reserved ids (padding/mask) never appear as
    # targets but get a bin anyway via the full-size lookup array.
    num_real = len(counts) - 2
    real_counts = counts[1:num_real + 1]
    item_bins = np.zeros(len(counts), dtype=np.int64)
    if args.tie_preserving:
        # items of equal count always share a bin: walk unique counts in
        # ascending order, close a bin when it reaches the target quantile size
        target = num_real / args.num_bins
        bin_of_count, current_bin, filled = {}, 0, 0
        for value, n in zip(*np.unique(real_counts, return_counts=True)):
            if filled >= target * (current_bin + 1) and current_bin < args.num_bins - 1:
                current_bin += 1
            bin_of_count[value] = current_bin
            filled += n
        used = sorted(set(bin_of_count.values()))
        remap = {b: i for i, b in enumerate(used)}
        for idx, value in enumerate(real_counts):
            item_bins[idx + 1] = remap[bin_of_count[value]]
        args.num_bins = len(used)
    else:
        order = np.lexsort((np.arange(1, num_real + 1), real_counts))
        for rank, idx in enumerate(order):
            item_bins[idx + 1] = min(rank * args.num_bins // num_real, args.num_bins - 1)

    dataset = BaseDataset.create_from_config(config['dataset'])
    _, validation_sampler, test_sampler = dataset.get_samplers()
    sampler = validation_sampler if args.split == 'validation' else test_sampler
    dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'], dataset=sampler, **dataset.meta,
    )
    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    hits, totals, user_recalls = collect_per_user(
        model, dataloader, args.k, item_bins, args.num_bins,
    )

    base_hits = None
    if args.baseline_checkpoint:
        base_config_path = args.baseline_params or args.params
        with open(base_config_path) as f:
            base_config = json.load(f)
        base_model = BaseModel.create_from_config(
            base_config['model'], **dataset.meta,
        ).to(DEVICE)
        base_ckpt = torch.load(args.baseline_checkpoint, map_location=DEVICE)
        base_model.load_state_dict(
            base_ckpt.get('model_state_dict', base_ckpt),
        )
        base_hits, base_totals, base_user_recalls = collect_per_user(
            base_model, dataloader, args.k, item_bins, args.num_bins,
        )
        assert np.array_equal(totals, base_totals), 'user/target mismatch'

    rng = np.random.default_rng(0)
    num_users = hits.shape[0]
    samples = np.empty((args.bootstrap, args.num_bins))
    diff_samples = np.empty_like(samples) if base_hits is not None else None
    for b in range(args.bootstrap):
        idx = rng.integers(0, num_users, num_users)
        tot = np.maximum(totals[idx].sum(0), 1)
        samples[b] = hits[idx].sum(0) / tot
        if diff_samples is not None:
            # paired: the SAME resampled users in both arms
            diff_samples[b] = samples[b] - base_hits[idx].sum(0) / tot
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)

    bin_recall = hits.sum(0) / np.maximum(totals.sum(0), 1)
    bin_counts = [counts[1:num_real + 1][item_bins[1:num_real + 1] == b]
                  for b in range(args.num_bins)]

    print(f'macro recall@{args.k} (tensorboard convention): {user_recalls.mean():.4f}  '
          f'[{num_users} users, {int(totals.sum())} target events]')
    print('CI: cluster bootstrap over test users, conditional on ONE trained '
          'seed (no seed variability included)')
    print(f'{"bin":>3} {"train count":>13} {"items":>7} {"events":>8} {"recall":>8}  95% CI')
    rows = []
    for b in range(args.num_bins):
        crange = f'{int(bin_counts[b].min())}-{int(bin_counts[b].max())}'
        n_events = int(totals.sum(0)[b])
        print(f'{b:>3} {crange:>13} {len(bin_counts[b]):>7} {n_events:>8} '
              f'{bin_recall[b]:>8.4f}  [{lo[b]:.4f}, {hi[b]:.4f}]')
        rows.append({
            'bin': b, 'count_range': crange, 'items': len(bin_counts[b]),
            'events': n_events,
            'recall': float(bin_recall[b]), 'ci': [float(lo[b]), float(hi[b])],
        })

    if base_hits is not None:
        dlo, dhi = np.percentile(diff_samples, [2.5, 97.5], axis=0)
        base_recall = base_hits.sum(0) / np.maximum(totals.sum(0), 1)
        print(f'\npaired differences (main - baseline), same users both arms; '
              f'baseline macro {base_user_recalls.mean():.4f}:')
        print(f'{"bin":>3} {"train count":>13} {"diff":>9}  95% CI{"":>12}sig')
        for b, row in enumerate(rows):
            d = bin_recall[b] - base_recall[b]
            sig = '*' if dlo[b] > 0 or dhi[b] < 0 else ''
            print(f'{b:>3} {row["count_range"]:>13} {d:>+9.4f}  '
                  f'[{dlo[b]:+.4f}, {dhi[b]:+.4f}]  {sig}')
            row['diff'] = float(d)
            row['diff_ci'] = [float(dlo[b]), float(dhi[b])]

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint, 'split': args.split, 'k': args.k,
                'baseline_checkpoint': args.baseline_checkpoint,
                'tie_preserving': bool(args.tie_preserving),
                'ci_note': 'user cluster bootstrap, conditional on one seed',
                'macro_recall': float(user_recalls.mean()), 'bins': rows,
            }, f, indent=2)


if __name__ == '__main__':
    main()
