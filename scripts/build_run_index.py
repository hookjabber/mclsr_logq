"""Build RUN_INDEX.md: every training run of the study, labelled, with numbers and links.

Sources: results/explor/explor_summary_*.txt (scripts/summarize_runs.py output over TensorBoard logs:
single-seed exploratory runs, test read at the validation-best step), results/confirm/*.json
(multi-seed confirmatory reports, test read once), configs/train/<dir>/<arm>.json.

    python scripts/build_run_index.py
"""

import glob
import json
import os
import re
import statistics
from collections import defaultdict

DATASETS = [
    ('clothing64', 'Clothing', ['configs/train/clothing64', 'configs/train/grid']),
    ('beauty', 'Beauty', ['configs/train/beauty']),
    ('toys', 'CDs & Vinyl ("Toys")', ['configs/train/toys']),
]

RUN_RE = re.compile(
    r'^(?P<run>\S+?)_(?P<ts>20\d\d-\d\d-\d\dT[\d:]+)\s+(?P<val>[\d.]+) @ (?P<vstep>\d+)\s+(?P<ndcg>[\d.]+|-)(?: @ \d+)?\s+(?P<rec>[\d.]+|-)(?: @ \d+)?\s+(?P<last>\d+)\s*$'
)


def describe(arm):
    """Human label from the arm identifier (the naming is systematic)."""
    base = {
        '01_orig': 'MCLSR without graph; in-batch sampled softmax, no correction (λ=0)',
        '02_logq_downstream': 'MCLSR without graph; in-batch + logQ on L_P (λ=1)',
        '03_graph': 'full model: user–item graph + L_IL; logQ on L_P',
        '04_graph_logq_lil': 'as 03, logQ also on L_IL',
        '05_full_baseline': 'paper model: graph + L_IL + L_UC + L_IC (γ=0.05); logQ on L_P',
        '05_full_uw': 'paper model with learned (uncertainty) loss weights',
        '06_full_logq_ucic': 'paper model, logQ also on L_UC / L_IC',
        '07_full_logq_all': 'paper model, logQ on every loss',
        '09_item_only': 'graph + L_IC only (no L_IL, no L_UC)',
        '10_item_only_logq': 'graph + L_IC with logQ',
        '10_item_only_logq_ctxq_v2': 'graph + L_IC with logQ from context-inclusion counts',
        '14_full_softmax': 'exact full-catalogue softmax on L_P (no sampling), no graph',
        '14_full_softmax_graph': 'exact full-catalogue softmax on L_P, with graph + L_IL',
        '17_user_only_full_matched': 'graph + full-catalogue matched L_UC only',
        '18_item_only_full_matched': 'graph + full-catalogue matched L_IC only',
        '03_graph_l00': 'graph + L_IL without the correction (λ=0) — factorial cell',
        '03_graph_il00': 'graph without L_IL (β=0)',
        '03_graph_paper_faithful': '03 with shared projector, cosine L_IL and the paper scheme',
        '03_graph_sharedproj': '03 with a shared L_IL projector',
        '03_graph_corrected': "03 with the RecSys'25 corrected form on L_P",
        '02_logq_corrected': "02 with the RecSys'25 corrected form (positive out of the denominator, sg(1−P̂) weight)",
        '02_logq_pos': '02 with the standard form (positive corrected too)',
        '02_logq_downstream_loo': '02 with leave-own-out q′',
        '02_logq_targetq': '02 with q from target counts',
        '02_mns128': '02 + 128 shared uniform negatives (MNS), mixture proposal',
        '02_mns128_std': 'MNS, standard form',
        '02_mns128_corr': 'MNS, corrected form',
        '01_nousermask': '01 without the same-user mask',
        '02_logq_nousermask': '02 without the same-user mask',
        '01_uniform127': 'no graph; 127 uniform negatives per query, no correction',
        '01_uniform1280': 'no graph; 1280 uniform negatives per query (as in the MCLSR paper)',
        '01_popular127': 'no graph; 127 popularity-sampled negatives',
        '02_popular127_logq': 'no graph; 127 popularity-sampled negatives + logQ',
        '04_graph_logq_lil_uniformq': '04 with a constant user-count table (pure margin, no popularity information)',
        '04_graph_logq_lil_centered': '04 with centred log q on L_IL',
        '05_minus_uc': 'paper model minus the user–user graph loss L_UC',
        '05_minus_ic': 'paper model minus the item–item graph loss L_IC',
        '05_minus_il': 'paper model minus the alignment loss L_IL',
        '03_graph_mean': '03 with the LightGCN layer mean (layers 0..2 averaged) instead of the last layer',
        '05_full_mean': 'paper model with the LightGCN layer mean instead of the last layer',
        '03_graph_depth0': '03 with no graph propagation (raw user / item embeddings feed the general-interest branch)',
        '03_graph_depth1': '03 with one propagation layer',
        '03_graph_gd0': '03 without graph edge dropout', '03_graph_mean_gd0': '03 with the layer mean and no edge dropout',
        '05_full_l00': 'paper model (graph + L_IL + L_UC + L_IC) without the correction (λ=0)',
        '01_orig_det': '01 with deterministic kernels (reproducibility check)',
        '01_orig_det_b': '01 with deterministic kernels, second run',
        '01_bs256': '01 with batch size 256',
        '01_bs512': '01 with batch size 512',
        '02_logq_bs256': '02 with batch size 256',
        '02_logq_bs512': '02 with batch size 512',
        '02_logq_l025': '02 with λ=0.25',
        '02_logq_l05': '02 with λ=0.5',
        '02_logq_l075': '02 with λ=0.75',
        '02_logq_l15': '02 with λ=1.5',
        '02_cosine_t01': '02 with cosine scores on L_P, τ=0.1',
        '02_cosine_t05': '02 with cosine scores on L_P, τ=0.5',
        '02_cosine_t1': '02 with cosine scores on L_P, τ=1',
        '03_graph_cosine': '03 with cosine L_IL',
        '03_graph_euclid': '03 with squared-euclidean L_IL',
        '03_graph_bxb': '03, L_IL over the batch×batch pool (August variant, see RESULTS.md Part B)',
        '03_graph_bxb_w05': '03 batch×batch, γ=0.5',
        '03_graph_paper_w05': '03 paper-faithful with γ=0.5',
        '03_graph_shared_proj': '03 with a shared L_IL projector',
        '04_graph_logq_lil_l00': '04 with λ_IL=0 (control)',
        '04_graph_logq_lil_l025': '04 with λ_IL=0.25',
        '04_graph_logq_lil_l05': '04 with λ_IL=0.5',
        '04_graph_logq_lil_l01': '04 with λ_IL=0.1',
        '04_graph_logq_lil_l03': '04 with λ_IL=0.3',
        '04_graph_logq_lil_pos': '04 with the positive corrected too (standard form) on L_IL',
        '04_graph_logq_lil_loo': '04 with leave-own-out q′ on L_IL',
        '04_graph_logq_lil_nomask': '04 without false-negative masking on L_IL',
        '04_graph_logq_lil_det': '04 with deterministic kernels',
        '04_graph_logq_lil_cosine': '04 with cosine L_IL',
        '05_full_baseline_g01': 'paper model with γ=0.1',
        '06_full_logq_ucic_g01': 'paper model, logQ on L_UC / L_IC, γ=0.1',
        '09_item_only_cosine': 'graph + cosine L_IC only',
        '10_item_only_logq_cosine': 'graph + cosine L_IC with logQ',
        '10_item_only_logq_ctxq': 'graph + L_IC with logQ from context-inclusion counts',
        '15_user_only_full': 'graph + full-catalogue L_UC only (unmatched)',
        '16_item_only_full': 'graph + full-catalogue L_IC only (unmatched)',
        '11_user_only': 'graph + L_UC only',
        '12_user_only_logq': 'graph + L_UC with logQ',
        '13_user_only_logq_nomask': 'graph + L_UC with logQ, no false-negative mask',
        '05_full_w01': 'paper model with γ=0.1',
        '05_full_w02': 'paper model with γ=0.2',
        '05_full_w05': 'paper model with γ=0.5',
        '03_graph_il05': '03 with L_IL weight β=0.5',
        '03_graph_il20': '03 with β=2',
        '03_graph_il025': '03 with β=0.25',
        '03_graph_a025': '03 with interest mix α=0.25',
        '03_graph_a075': '03 with α=0.75',
        '03_graph_a09': '03 with α=0.9',
        '03_graph_euclid_t05': '03 with squared-euclidean L_IL, τ=0.5',
        '03_graph_euclid_t1': '03 with squared-euclidean L_IL, τ=1',
        '03_graph_euclid_t20': '03 with squared-euclidean L_IL, τ=2',
        '03_graph_cosine_t01': '03 with cosine L_IL, τ=0.1',
        '03_graph_cosine_t02': '03 with cosine L_IL, τ=0.2',
        '03_graph_cosine_t05': '03 with cosine L_IL, τ=0.5',
        '03_graph_cosine_t10': '03 with cosine L_IL, τ=1',
        '03_graph_cosine_t1': '03 with cosine L_IL, τ=1',
        '04_graph_logq_lil_euclid_t05': '04 with squared-euclidean L_IL, τ=0.5',
        '04_graph_logq_lil_euclid_t10': '04 with squared-euclidean L_IL, τ=1',
        '04_graph_logq_lil_euclid_t20': '04 with squared-euclidean L_IL, τ=2',
        '04_graph_logq_lil_cosine_t05': '04 with cosine L_IL, τ=0.5',
        'sasrec_inbatch_l00': 'SASRec, in-batch sampled softmax, no correction',
        'sasrec_inbatch_logq': 'SASRec, in-batch + logQ',
        'sasrec_baseline': 'SASRec, classic BCE with one negative',
        'sasrec_inbatch_corrected': "SASRec, RecSys'25 corrected form",
        'sasrec_inbatch_l00_umask': 'SASRec λ=0 with the same-user mask',
        'sasrec_inbatch_logq_umask': 'SASRec λ=1 with the same-user mask',
        'sasrec_ladder_l00': 'SASRec λ=0 on the MCLSR prefix ladder (last-position query)',
        'sasrec_ladder_logq': 'SASRec λ=1 on the prefix ladder',
    }
    if arm in base:
        return base[arm]
    root = None
    for key in sorted(base, key=len, reverse=True):
        if arm.startswith(key + '_'):
            root, suffix = key, arm[len(key) + 1 :]
            break
    if root is None:
        return arm
    s = suffix
    rules = [
        (r'^euclid_t(\d+)$', lambda g: f'squared-euclidean L_IL, τ={int(g[0]) / 10 if int(g[0]) >= 5 else g[0]}'),
        (r'^cosine_t(\d+)$', lambda g: f'cosine L_IL, τ={int(g[0]) / 10 if len(g[0]) == 2 else g[0]}'),
        (r'^il(\d+)$', lambda g: f'L_IL weight β={int(g[0]) / 10}'),
        (r'^a0?(\d+)$', lambda g: f'interest mix α=0.{g[0]}'),
        (r'^w0?(\d+)$', lambda g: f'feature-level weight γ=0.{g[0]}'),
        (
            r'^l0?(\d+)$',
            lambda g: f'λ on the corrected loss = {int(g[0]) / (10 if len(g[0]) == 2 else 100) if g[0] != "15" else 1.5}',
        ),
        (r'^bs(\d+)$', lambda g: f'batch size {g[0]}'),
        (r'^e64$', lambda g: 'validation every 64 steps (exact test at val-best)'),
    ]
    for pat, fn in rules:
        mm = re.match(pat, s)
        if mm:
            return f'{base[root]}; {fn(mm.groups())}'
    return f'{base[root]}; variant {s}'


def load_explor():
    rows = defaultdict(list)  # (dataset_key, arm) -> list of dicts
    for path in glob.glob('results/explor/explor_summary_*.txt'):
        for line in open(path):
            line = re.sub(r'\s+', ' ', line.strip())
            m = RUN_RE.match(line)
            if not m or '_confirm_seed' in m['run']:
                continue
            run = m['run']
            m2 = re.match(
                r'^(?:(?P<pfx>beauty|toys)_)?(?P<fam>mclsr_grid|sasrec)_(?P<arm>.+?)(?:_(?:Beauty|Toys|Clothing)(?:_.*)?)?$',
                run,
            )
            if not m2:
                continue
            key = m2['pfx'] or 'clothing64'
            arm = m2['arm'] if m2['fam'] == 'mclsr_grid' else 'sasrec_' + m2['arm']
            arm = re.sub(r'_e64$', '', arm)
            rows[(key, arm)].append(
                {
                    'ts': m['ts'],
                    'val': float(m['val']),
                    'ndcg': None if m['ndcg'] == '-' else float(m['ndcg']),
                    'rec': None if m['rec'] == '-' else float(m['rec']),
                    'last': int(m['last']),
                }
            )
    return rows


def load_confirm():
    rows = defaultdict(dict)
    for path in glob.glob('results/confirm/*_seed*.json'):
        name = os.path.basename(path)
        m = re.match(r'(beauty|toys|clothing64)_(.+)_seed(\d+)\.json', name)
        if not m:
            continue
        r = json.load(open(path))['results']
        rows[(m[1], m[2])][int(m[3])] = (
            r['validation/ndcg@20']['test']['ndcg@20'],
            r['validation/recall@1000']['test']['recall@1000'],
            name,
        )
    return rows


def fmt_seeds(d):
    nd = [v[0] for _, v in sorted(d.items())]
    rc = [v[1] for _, v in sorted(d.items())]

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return f'{statistics.mean(nd):.4f} ± {sd(nd):.4f} / {statistics.mean(rc):.4f} ± {sd(rc):.4f}', ','.join(
        str(s) for s in sorted(d)
    )


def config_link(dirs, arm):
    for d in dirs:
        p = f'{d}/{arm}.json'
        if os.path.exists(p):
            return f'[{arm}]({p})'
    return arm


def main():
    explor, confirm = load_explor(), load_confirm()
    out = [
        '# Run index — every training run of the logQ study, labelled',
        '',
        'One row per arm and dataset. **Seeds** columns: confirmatory runs (`scripts/train_confirmatory.py`, test read once on the validation-selected checkpoint; reports in `results/confirm/`, mean ± std over the listed seeds). **Single run** columns: exploratory run with seed 42 (test taken at the validation-best step from the TensorBoard log; approximate to ±1 evaluation interval). Metrics: test ndcg@20 / recall@1000 over the full catalogue. Figures and headline tables: `RESULTS.md`.',
        '',
    ]
    total = 0
    for key, title, dirs in DATASETS:
        arms = sorted({a for (k, a) in list(explor) + list(confirm) if k == key})
        out += [
            f'## {title} — {len(arms)} arms',
            '',
            '| arm (config) | what it is | seeds: ndcg@20 / recall@1000 | seeds | single run (seed 42): ndcg@20 / recall@1000 | runs |',
            '|---|---|---|---|---|---|',
        ]
        for arm in arms:
            c = confirm.get((key, arm))
            e = explor.get((key, arm), [])
            seeds_txt, seeds_list = fmt_seeds(c) if c else ('—', '—')
            if e:
                best = sorted(e, key=lambda r: r['ts'])[-1]
                single = (
                    f'{best["ndcg"]:.4f} / {best["rec"]:.4f}'
                    if best['ndcg'] is not None and best['rec'] is not None
                    else (f'{best["ndcg"]:.4f} / —' if best['ndcg'] is not None else '—')
                )
            else:
                single = '—'
            total += len(e) + (len(c) if c else 0)
            out.append(
                f'| {config_link(dirs, arm)} | {describe(arm)} | {seeds_txt} | {seeds_list} | {single} | {len(e) + (len(c) if c else 0)} |'
            )
        out.append('')
    out.append(
        f'Total training runs indexed: {total}. Re-run summaries: `python scripts/summarize_runs.py --pattern "<prefix>_*" --report eval/ndcg@20 eval/recall@1000` over `tensorboard_logs/`, and `python scripts/summarize_seeds.py --prefix <beauty|toys|clothing64>`.'
    )
    open('RUN_INDEX.md', 'w').write('\n'.join(out) + '\n')
    print(
        'arms per dataset:',
        {k: len({a for (kk, a) in list(explor) + list(confirm) if kk == k}) for k, _, _ in DATASETS},
        'runs',
        total,
    )


if __name__ == '__main__':
    main()
