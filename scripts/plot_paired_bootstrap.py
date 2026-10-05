"""Interval plot of the paired per-user comparisons (results/stats/<prefix>_paired_bootstrap.json).

python scripts/plot_paired_bootstrap.py            # writes assets/paired_bootstrap_beauty.png
"""

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    '02_logq_downstream:01_orig': 'logQ on L_P, no graph',
    '03_graph:03_graph_l00': 'logQ on L_P, with graph + L_IL',
    '05_full_baseline:05_full_l00': 'logQ on L_P, full model',
    'sasrec_inbatch_logq:sasrec_inbatch_l00': 'logQ on L_P, SASRec',
    '04_graph_logq_lil:03_graph': 'logQ also on L_IL',
    '14_full_softmax:02_logq_downstream': 'exact softmax vs in-batch + logQ',
    '05_full_baseline:03_graph': 'user–user / item–item graphs',
}
INK, MUTED, MARK, GRID = '#1f2933', '#616e7c', '#1f77b4', '#e4e7eb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='beauty')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    stats = json.load(open(f'results/stats/{args.prefix}_paired_bootstrap.json'))
    pairs = [p for p in LABELS if any(k.startswith(p + ':') for k in stats)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.55 * len(pairs) + 1.6), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, metric, title in zip(axes, ('ndcg@20', 'recall@1000'), ('Δ test ndcg@20', 'Δ test recall@1000')):
        ys = range(len(pairs))[::-1]
        for y, pair in zip(ys, pairs):
            r = stats[f'{pair}:{metric}']
            lo, hi, m, p = r['ci95'][0], r['ci95'][1], r['mean'], r['p_sign_flip']
            ax.plot([lo, hi], [y, y], color=MARK, linewidth=2, solid_capstyle='round', zorder=2)
            ax.plot([m], [y], marker='o', markersize=8, color=MARK, zorder=3)
            ax.annotate(
                f'{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   p {p:.4f}'
                if p >= 0.001
                else f'{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   p < 0.001',
                (hi, y),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                fontsize=8.5,
                color=MUTED,
            )
        ax.axvline(0, color=INK, linewidth=1, zorder=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([LABELS[p] for p in pairs], fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc='left')
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(axis='x', labelsize=8.5, colors=MUTED)
        ax.tick_params(axis='y', length=0)
        lo_all = min(stats[f'{p}:{metric}']['ci95'][0] for p in pairs)
        hi_all = max(stats[f'{p}:{metric}']['ci95'][1] for p in pairs)
        span = hi_all - lo_all
        ax.set_xlim(min(lo_all, 0) - 0.08 * span, hi_all + 1.75 * span)
    fig.suptitle(
        f'Paired per-user differences, {args.prefix.capitalize()} test users, 3 seeds pooled; '
        '95 % user-bootstrap interval, sign-flip p',
        fontsize=10,
        color=MUTED,
        x=0.01,
        ha='left',
    )
    fig.subplots_adjust(left=0.2, right=0.995, top=0.86, bottom=0.1, wspace=0.08)
    out = args.output or f'assets/paired_bootstrap_{args.prefix}.png'
    fig.savefig(out, dpi=160, facecolor='white')
    print('written', out)


if __name__ == '__main__':
    main()
