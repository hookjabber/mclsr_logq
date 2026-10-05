"""End-to-end smoke test of the whole pipeline on a synthetic dataset, CPU only, ~20 s.

Raw sequences -> ladder / eval files and a count table (the formats of ``scripts/prepare_dataset.py``
and ``scripts/generate_item_counts.py``) -> the confirmatory runner (graphs built from the files,
full MCLSR with all four losses and the logQ correction, validation-selected checkpoint, one test
read, JSON report with hashes) -> the report is checked for shape and sanity, the run is repeated
with the same seed and must reproduce the same test numbers.
"""

import copy
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MAX_LEN_TRAIN = 20


def _synthetic_dataset(data_dir, num_users=90, num_items=40, seed=0):
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** 0.6 for i in range(num_items)]  # popularity skew like real catalogues
    seqs = []
    for uid in range(1, num_users + 1):
        length = rng.randint(5, 14)
        items = []
        while len(items) < length:
            item = rng.choices(range(1, num_items + 1), weights=weights)[0]
            if item not in items:
                items.append(item)
        seqs.append((uid, items))
    users = [u for u, _ in seqs]
    rng.shuffle(users)
    n_train, n_valid = int(num_users * 0.8), int(num_users * 0.1)
    train_u, valid_u = set(users[:n_train]), set(users[n_train : n_train + n_valid])
    os.makedirs(data_dir, exist_ok=True)
    fh = {
        n: open(os.path.join(data_dir, n + '.txt'), 'w')
        for n in ('train_mclsr', 'train_sasrec', 'valid_history', 'valid_target', 'test_history', 'test_target')
    }
    counts = np.zeros(num_items + 2, dtype=np.float32)
    for uid, items in seqs:
        if uid in train_u:
            crop = items[-MAX_LEN_TRAIN:]
            fh['train_sasrec'].write(' '.join(map(str, [uid] + crop)) + '\n')
            for i in range(1, len(crop)):
                fh['train_mclsr'].write(' '.join(map(str, [uid] + crop[:i] + [crop[i]])) + '\n')
            for item in crop:
                counts[item] += 1
        else:
            si = int(len(items) * 0.8)
            kind = 'valid' if uid in valid_u else 'test'
            fh[kind + '_history'].write(' '.join(map(str, [uid] + items[:si])) + '\n')
            fh[kind + '_target'].write(' '.join(map(str, [uid] + items[si:])) + '\n')
    for f in fh.values():
        f.close()
    counts[counts == 0] = 1.0
    with open(os.path.join(data_dir, 'item_counts.pkl'), 'wb') as f:
        pickle.dump(counts, f)
    return num_items


def _config(data_root, name, num_items, steps):
    with open(os.path.join(ROOT, 'configs/train/beauty/05_full_baseline.json')) as f:
        cfg = json.load(f)
    cfg['experiment_name'] = 'e2e_synthetic'
    cfg['dataset']['graph_dir_path'] = os.path.join(data_root, name)
    cfg['dataset']['dataset']['path_to_data_dir'] = data_root
    cfg['dataset']['dataset']['name'] = name
    cfg['dataloader']['train']['batch_size'] = 16
    cfg['dataloader']['validation']['batch_size'] = 16
    cfg['model']['eval_top_k'] = 20  # the catalogue is tiny: metrics only up to @20
    cfg['loss']['losses'][0]['path_to_item_counts'] = os.path.join(data_root, name, 'item_counts.pkl')
    small = {
        k: v
        for k, v in {
            'ndcg@5': {'type': 'mclsr-ndcg', 'k': 5},
            'ndcg@20': {'type': 'mclsr-ndcg', 'k': 20},
            'recall@20': {'type': 'mclsr-recall', 'k': 20},
            'hit@20': {'type': 'mclsr-hit', 'k': 20},
        }.items()
    }
    for cb in cfg['callback']['callbacks']:
        if cb['type'] in ('validation', 'eval'):
            cb['metrics'] = copy.deepcopy(small)
            cb['on_step'] = 8
    cfg['train_steps_num'] = steps
    cfg['epochs_threshold'] = 100
    return cfg


def _run(workdir, cfg_path, report_path, seed):
    os.makedirs(workdir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src'), CUDA_VISIBLE_DEVICES='')
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, 'scripts/train_confirmatory.py'),
            '--params',
            cfg_path,
            '--seed',
            str(seed),
            '--select',
            'validation/ndcg@20',
            '--output',
            report_path,
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    with open(report_path) as f:
        return json.load(f)


def test_pipeline_end_to_end(tmp_path):
    data_root = str(tmp_path / 'data')
    num_items = _synthetic_dataset(os.path.join(data_root, 'Tiny'))
    cfg = _config(data_root, 'Tiny', num_items, steps=48)
    cfg_path = str(tmp_path / 'e2e.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    workdir = str(tmp_path / 'run')

    report = _run(workdir, cfg_path, str(tmp_path / 'report1.json'), seed=1)

    # graphs were built from the files (train-only), with the user-user / item-item neighbourhoods
    for name in ('general_graph_trainOnly.npz', 'user_graph_k50_trainOnly.npz', 'item_graph_k50_trainOnly.npz'):
        assert os.path.exists(os.path.join(data_root, 'Tiny', name)), name
    # the report: one validation-selected checkpoint, scored once on test, hashes present
    best = report['results']['validation/ndcg@20']
    assert 0.0 <= best['validation_value'] <= 1.0 and best['step'] <= 48
    for metric in ('ndcg@5', 'ndcg@20', 'recall@20', 'hit@20'):
        assert 0.0 <= best['test'][metric] <= 1.0, metric
    assert best['test']['ndcg@5'] <= best['test']['ndcg@20'] <= best['test']['hit@20']
    assert report['step_limit'] == 48 and report['seed'] == 1 and 'torch_version' in report
    checkpoint = os.path.join(workdir, best['checkpoint'])
    assert os.path.exists(checkpoint)
    with open(checkpoint, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == best['checkpoint_sha256']
    assert report['params_sha256'] == hashlib.sha256(open(cfg_path, 'rb').read()).hexdigest()

    # the same seed reproduces the same selection and the same test numbers
    again = _run(str(tmp_path / 'run2'), cfg_path, str(tmp_path / 'report2.json'), seed=1)
    assert again['results']['validation/ndcg@20']['step'] == best['step']
    for metric in ('ndcg@20', 'recall@20'):
        assert abs(again['results']['validation/ndcg@20']['test'][metric] - best['test'][metric]) < 1e-6, metric
