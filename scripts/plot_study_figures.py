"""Figures for the multi-seed logQ study (RESULTS.md Part A). Reads results/confirm/*.json and
results/deciles/*.json; single-seed sweep arms are quoted from the run log.

    python scripts/plot_study_figures.py  # writes assets/*.png
"""

import glob
import json
import os
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [('clothing64', 'Clothing'), ('beauty', 'Beauty'), ('toys', 'CDs & Vinyl')]
C = {
    'l0': '#9aa5b1',
    'l1': '#1f77b4',
    'exact': '#2ca02c',
    's0': '#d9c39a',
    's1': '#ff7f0e',
    'g': '#8c564b',
    'r': '#d62728',
}


def seeds(prefix, arm):
    rows = []
    for path in sorted(glob.glob(f'results/confirm/{prefix}_{arm}_seed*.json')):
        r = json.load(open(path))['results']
        rows.append((r['validation/ndcg@20']['test']['ndcg@20'], r['validation/recall@1000']['test']['recall@1000']))
    return rows


def ms(prefix, arm, idx):
    vals = [r[idx] for r in seeds(prefix, arm)]
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0) if vals else (np.nan, 0.0)


def fig_headline():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    arms = [
        ('01_orig', 'MCLSR, in-batch λ=0', C['l0']),
        ('02_logq_downstream', 'MCLSR, in-batch + logQ', C['l1']),
        ('14_full_softmax', 'MCLSR, exact softmax', C['exact']),
        ('sasrec_inbatch_l00', 'SASRec, in-batch λ=0', C['s0']),
        ('sasrec_inbatch_logq', 'SASRec, in-batch + logQ', C['s1']),
    ]
    width = 0.15
    for ax, idx, title in zip(axes, (0, 1), ('test ndcg@20', 'test recall@1000')):
        for j, (arm, label, color) in enumerate(arms):
            xs = np.arange(len(DATASETS)) + (j - 2) * width
            means, stds = zip(*[ms(p, arm, idx) for p, _ in DATASETS])
            ax.bar(xs, means, width, yerr=stds, capsize=2, color=color, label=label, edgecolor='none')
        for i, (p, _) in enumerate(DATASETS):
            for base, arm, j in (
                ('01_orig', '02_logq_downstream', 1),
                ('sasrec_inbatch_l00', 'sasrec_inbatch_logq', 4),
            ):
                b, a = ms(p, base, idx)[0], ms(p, arm, idx)[0]
                ax.annotate(
                    f'+{100 * (a / b - 1):.0f}%',
                    (i + (j - 2) * width, a),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                )
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([n for _, n in DATASETS])
        ax.set_title(title, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(
        'logQ correction on the retrieval loss: three datasets, two encoders (mean ± std over 3 seeds)', fontsize=11
    )
    fig.tight_layout()
    fig.savefig('assets/headline_three_datasets.png', dpi=150)
    plt.close(fig)


def fig_factorial():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, idx, title in zip(axes, (0, 1), ('test ndcg@20', 'test recall@1000')):
        for p, name, ls in (('beauty', 'Beauty', '-'), ('toys', 'CDs & Vinyl', '--')):
            for arms, label, color in (
                (('01_orig', '02_logq_downstream'), 'no graph', C['l1']),
                (('03_graph_l00', '03_graph'), 'graph + L_IL', C['g']),
            ):
                means, stds = zip(*[ms(p, a, idx) for a in arms])
                ax.errorbar([0, 1], means, yerr=stds, fmt='o' + ls, color=color, capsize=3, label=f'{name}, {label}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['λ = 0 (no correction)', 'λ = 1 (logQ on L_P)'])
        ax.set_title(title, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        'Graph × logQ factorial: the correction adds the same gain with and without the graph; the graph adds tail recall only',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig('assets/factorial_graph_logq.png', dpi=150)
    plt.close(fig)


def fig_lil():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    bars = [
        ('L_IL uncorrected\n(arm 03)', ms('beauty', '03_graph', 0), C['l1']),
        ('λ_IL = 0.1', (0.0565, 0), C['r']),
        ('λ_IL = 0.3', (0.0544, 0), C['r']),
        ('λ_IL = 1, real q\n(arm 04)', ms('beauty', '04_graph_logq_lil', 0), C['r']),
        ('λ_IL = 1, constant q\n(margin only)', ms('beauty', '04_graph_logq_lil_uniformq', 0), '#7f7f7f'),
    ]
    ax.bar(
        range(len(bars)), [b[1][0] for b in bars], yerr=[b[1][1] for b in bars], capsize=3, color=[b[2] for b in bars]
    )
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0.045, 0.060)
    ax.set_title('Beauty: correcting L_IL hurts; a constant-q correction (a margin) reproduces it', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('test ndcg@20')
    ax = axes[1]
    betas = [0, 0.25, 0.5, 1, 2]
    nd = [
        ms('beauty', '03_graph_il00', 0),
        (0.0569, 0),
        ms('beauty', '03_graph_il05', 0),
        ms('beauty', '03_graph', 0),
        (0.0532, 0),
    ]
    rc = [
        ms('beauty', '03_graph_il00', 1),
        (0.5454, 0),
        ms('beauty', '03_graph_il05', 1),
        ms('beauty', '03_graph', 1),
        (0.5151, 0),
    ]
    ax.errorbar(
        range(5),
        [v[0] for v in nd],
        yerr=[v[1] for v in nd],
        fmt='o-',
        color=C['l1'],
        capsize=3,
        label='test ndcg@20 (left)',
    )
    ax2 = ax.twinx()
    ax2.errorbar(
        range(5),
        [v[0] for v in rc],
        yerr=[v[1] for v in rc],
        fmt='s--',
        color=C['g'],
        capsize=3,
        label='test recall@1000 (right)',
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'β = {b}' for b in betas])
    ax.set_title('Beauty: weight β of L_IL — β = 0 loses the tail; plateau 0.25–1', fontsize=9)
    ax.spines[['top']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc='lower center')
    fig.tight_layout()
    fig.savefig('assets/lil_mechanism.png', dpi=150)
    plt.close(fig)


def fig_deciles():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for path, label, color, off in (
        ('results/deciles/beauty_02_vs_01.json', 'Beauty: logQ − none', C['l1'], -0.18),
        ('results/deciles/toys_02_vs_01.json', 'CDs & Vinyl: logQ − none', C['s1'], 0.18),
    ):
        d = json.load(open(path))
        bins = d['bins']
        xs = np.arange(len(bins)) + off
        diff = [b['diff'] for b in bins]
        err = np.array([[b['diff'] - b['diff_ci'][0] for b in bins], [b['diff_ci'][1] - b['diff'] for b in bins]])
        ax.bar(xs, diff, 0.36, yerr=err, capsize=2, color=color, label=label)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b['count_range'] for b in bins], fontsize=8, rotation=45)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('item popularity decile (train count range), tail → head')
    ax.set_ylabel('Δ recall@1000 (95 % bootstrap CI)')
    ax.set_title('The correction gains in the head of the catalogue', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    for s in (1, 2, 3):
        d = json.load(open(f'results/deciles/beauty_l05_vs_02_seed{s}.json'))
        bins = d['bins']
        ax.plot(range(len(bins)), [b['diff'] for b in bins], 'o-', label=f'seed {s}', alpha=0.8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b['count_range'] for b in bins], fontsize=8, rotation=45)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('item popularity decile (train count range), tail → head')
    ax.set_ylabel('Δ recall@1000, λ = 0.5 − λ = 1')
    ax.set_title('Beauty: λ = 0.5 moves recall from head to tail (and loses overall)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('assets/deciles_logq.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    fig_headline()
    fig_factorial()
    fig_lil()
    fig_deciles()
    print('figures written:', sorted(f for f in os.listdir('assets') if f.endswith('.png')))
