#!/usr/bin/env python
"""Summarize tensorboard runs under the honest selection protocol.

Default protocol `at-val-best`: pick the checkpoint step by the best value of
--select (a validation metric), then report every --report metric at its logged
event nearest to that step (test curves are logged on a sparser grid, so the
nearest event can be up to half an eval period away). This is the protocol the
tables in RESULTS.md use.

`--protocol max` reproduces the old behavior — an independent max per metric.
That is an optimistic bound useful only for curve inspection, never a result.

Usage:
    python scripts/summarize_runs.py                              # ndcg@20 selection
    python scripts/summarize_runs.py --select validation/recall@1000 \
        --report eval/recall@1000 eval/ndcg@20                    # candidate-gen table
    python scripts/summarize_runs.py --protocol max --pattern 'mclsr_grid_04*'
"""
import argparse
import fnmatch
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    return {tag: acc.Scalars(tag) for tag in acc.Tags().get('scalars', [])}


def last_step(scalars):
    steps = [events[-1].step for events in scalars.values() if events]
    return max(steps) if steps else None


def at_val_best(scalars, select, report):
    """(select_best, {report_tag: value at the event nearest to that step})"""
    val = scalars.get(select)
    if not val:
        return None, {}
    best = max(val, key=lambda e: e.value)
    row = {}
    for tag in report:
        events = scalars.get(tag)
        if events:
            nearest = min(events, key=lambda e: abs(e.step - best.step))
            row[tag] = (nearest.value, nearest.step)
    return (best.value, best.step), row


def independent_max(scalars, metrics):
    row = {}
    for tag in metrics:
        events = scalars.get(tag)
        if events:
            best = max(events, key=lambda e: e.value)
            row[tag] = (best.value, best.step)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', default='tensorboard_logs')
    parser.add_argument(
        '--pattern', nargs='+',
        default=['mclsr_grid_*', 'sasrec_*'],
    )
    parser.add_argument(
        '--protocol', choices=['at-val-best', 'max'], default='at-val-best',
    )
    parser.add_argument(
        '--select', default='validation/ndcg@20',
        help='checkpoint-selection metric (at-val-best protocol)',
    )
    parser.add_argument(
        '--report', nargs='+', default=['eval/ndcg@20'],
        help='metrics reported at the selected checkpoint (at-val-best protocol)',
    )
    parser.add_argument(
        '--metrics', nargs='+',
        default=['validation/ndcg@20', 'eval/ndcg@20'],
        help='metrics for the legacy max protocol',
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
    if args.protocol == 'at-val-best':
        columns = [f'{args.select} best (step)'] + [
            f'{tag} @ ckpt' for tag in args.report
        ]
    else:
        columns = [f'{tag} max (step)' for tag in args.metrics]
    header = 'run'.ljust(name_w) + ''.join(c.rjust(34) for c in columns) + 'last step'.rjust(12)
    print(header)
    print('-' * len(header))

    for d in run_dirs:
        try:
            scalars = load_scalars(os.path.join(args.logs, d))
        except Exception as exc:
            print(d.ljust(name_w) + f'  ERROR: {exc}')
            continue
        cells = []
        if args.protocol == 'at-val-best':
            best, row = at_val_best(scalars, args.select, args.report)
            cells.append('-' if best is None else f'{best[0]:.4f} @ {best[1]}')
            for tag in args.report:
                if tag in row:
                    value, step = row[tag]
                    cells.append(f'{value:.4f} @ {step}')
                else:
                    cells.append('-')
        else:
            row = independent_max(scalars, args.metrics)
            for tag in args.metrics:
                if tag in row:
                    value, step = row[tag]
                    cells.append(f'{value:.4f} @ {step}')
                else:
                    cells.append('-')
        last = last_step(scalars)
        print(
            d.ljust(name_w)
            + ''.join(c.rjust(34) for c in cells)
            + str(last if last is not None else '-').rjust(12)
        )


if __name__ == '__main__':
    main()
