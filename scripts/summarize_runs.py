#!/usr/bin/env python
"""Summarize best metrics across tensorboard runs.

Usage:
    python scripts/summarize_runs.py                              # all grid + sasrec runs
    python scripts/summarize_runs.py --pattern 'mclsr_grid_04*'   # one family
    python scripts/summarize_runs.py --pattern 'mclsr_grid_1[123]*' 'mclsr_grid_02*'

Prints best validation/eval ndcg@20 (with the step it peaked at) and the last
logged step, so truncated/running runs are visible at a glance.
"""
import argparse
import fnmatch
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(run_dir, metrics):
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    tags = set(acc.Tags().get('scalars', []))
    row = {}
    last_step = None
    for key in metrics:
        if key in tags:
            events = acc.Scalars(key)
            if events:
                best = max(events, key=lambda e: e.value)
                row[key] = (best.value, best.step)
                last_step = max(last_step or 0, events[-1].step)
    return row, last_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', default='tensorboard_logs')
    parser.add_argument(
        '--pattern', nargs='+',
        default=['mclsr_grid_*', 'sasrec_*'],
    )
    parser.add_argument(
        '--metrics', nargs='+',
        default=['validation/ndcg@20', 'eval/ndcg@20'],
    )
    args = parser.parse_args()

    run_dirs = sorted(
        d for d in os.listdir(args.logs)
        if os.path.isdir(os.path.join(args.logs, d))
        and any(fnmatch.fnmatch(d, p) for p in args.pattern)
    )

    if not run_dirs:
        print(f'No runs in {args.logs} matching {args.pattern}')
        return

    name_w = max(len(d) for d in run_dirs) + 2
    header = 'run'.ljust(name_w) + ''.join(
        f'{m.split("/")[0]} best (step)'.rjust(28) for m in args.metrics
    ) + 'last step'.rjust(12)
    print(header)
    print('-' * len(header))

    for d in run_dirs:
        try:
            row, last_step = summarize(os.path.join(args.logs, d), args.metrics)
        except Exception as exc:
            print(d.ljust(name_w) + f'  ERROR: {exc}')
            continue
        cells = ''
        for m in args.metrics:
            if m in row:
                value, step = row[m]
                cells += f'{value:.4f} @ {step}'.rjust(28)
            else:
                cells += '-'.rjust(28)
        cells += str(last_step if last_step is not None else '-').rjust(12)
        print(d.ljust(name_w) + cells)


if __name__ == '__main__':
    main()
