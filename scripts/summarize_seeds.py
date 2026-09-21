"""Multi-seed summary of confirmatory reports (results/confirm/<prefix>_<arm>_seed<N>.json).

    python scripts/summarize_seeds.py --prefix beauty
    python scripts/summarize_seeds.py --prefix toys --pairs 02_logq_downstream:01_orig 04_graph_logq_lil:03_graph

Prints mean ± std and per-seed test values (ndcg@20 from the ndcg-selected checkpoint,
recall@1000 from the recall-selected one) and paired per-seed differences for the
requested arm pairs.
"""

import argparse
import glob
import json
import re
import statistics

DEFAULT_PAIRS = [
    '02_logq_downstream:01_orig',
    '04_graph_logq_lil:03_graph',
    '03_graph:03_graph_l00',
    '03_graph:02_logq_downstream',
    '14_full_softmax:02_logq_downstream',
    'sasrec_inbatch_logq:sasrec_inbatch_l00',
]


def load(prefix, confirm_dir):
    rows = {}
    for path in sorted(glob.glob(f'{confirm_dir}/{prefix}_*_seed*.json')):
        match = re.match(rf'.*/{prefix}_(.+)_seed(\d+)\.json', path)
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        results = json.load(open(path))['results']
        ndcg = results['validation/ndcg@20']
        recall = results['validation/recall@1000']
        rows.setdefault(arm, {})[seed] = {
            'ndcg@20': ndcg['test']['ndcg@20'],
            'recall@1000': recall['test']['recall@1000'],
            'val_ndcg@20': ndcg['validation_value'],
            'epoch': ndcg['epoch'],
        }
    return rows


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True, help='beauty | toys | clothing64')
    parser.add_argument('--confirm-dir', default='results/confirm')
    parser.add_argument('--pairs', nargs='*', default=DEFAULT_PAIRS, help='arm:baseline pairs')
    args = parser.parse_args()

    rows = load(args.prefix, args.confirm_dir)
    print(f'### {args.prefix}')
    print(
        '| arm | seeds | test ndcg@20 mean±std (per seed) | test recall@1000 mean±std (per seed) | val ndcg@20 | peak epoch |'
    )
    print('|---|---|---|---|---|---|')
    for arm in sorted(rows):
        seeds = sorted(rows[arm])
        ndcg = [rows[arm][s]['ndcg@20'] for s in seeds]
        recall = [rows[arm][s]['recall@1000'] for s in seeds]
        val = [rows[arm][s]['val_ndcg@20'] for s in seeds]
        epochs = [rows[arm][s]['epoch'] for s in seeds]
        (mn, sn), (mr, sr) = mean_std(ndcg), mean_std(recall)
        print(
            '| %s | %s | %.4f±%.4f (%s) | %.4f±%.4f (%s) | %.4f | %s |'
            % (
                arm,
                ','.join(map(str, seeds)),
                mn,
                sn,
                ', '.join('%.4f' % x for x in ndcg),
                mr,
                sr,
                ', '.join('%.4f' % x for x in recall),
                statistics.mean(val),
                epochs,
            )
        )
    print()
    for pair in args.pairs:
        arm, base = pair.split(':')
        common = sorted(set(rows.get(arm, {})) & set(rows.get(base, {})))
        if not common:
            continue
        d_ndcg = [rows[arm][s]['ndcg@20'] - rows[base][s]['ndcg@20'] for s in common]
        d_recall = [rows[arm][s]['recall@1000'] - rows[base][s]['recall@1000'] for s in common]
        print(
            '%s − %s (seeds %s): Δndcg@20 %s (mean %+.4f); Δrecall@1000 %s (mean %+.4f)'
            % (
                arm,
                base,
                common,
                ', '.join('%+.4f' % x for x in d_ndcg),
                statistics.mean(d_ndcg),
                ', '.join('%+.4f' % x for x in d_recall),
                statistics.mean(d_recall),
            )
        )


if __name__ == '__main__':
    main()
