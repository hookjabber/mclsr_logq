"""Freeze a five-arm legacy/windowed-v2 retrieval-logQ study without changing old data."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np

from prepare_dataset import sequences_from_csv, sequences_from_seq_file, split_users, write_files

FILES = tuple(
    f'{name}.txt'
    for name in ('train_mclsr', 'train_sasrec', 'valid_history', 'valid_target', 'test_history', 'test_target')
)
ARMS = ('legacy_l00', 'v2_l00', 'legacy_target_l10', 'v2_target_l10', 'legacy_proxy_l10')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def source_hashes(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for directory in ('src', 'scripts')
        for path in sorted((root / directory).rglob('*.py'))
    }


def write_json(path, content):
    with Path(path).open('x') as destination:
        json.dump(content, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write('\n')


def rows(path):
    with Path(path).open() as source:
        for line in source:
            values = [int(value) for value in line.split()]
            if values:
                yield values[0], values[1:]


def verify_evaluation(legacy, versioned):
    manifest = json.loads((Path(versioned) / 'preprocessing.json').read_text())
    users = {new: int(old) for old, new in manifest['user_id_mapping'].items()}
    items = {new: int(old) for old, new in manifest['item_id_mapping'].items()}
    for filename in FILES[2:]:
        try:
            before = list(rows(Path(legacy) / filename))
            after = [
                (users[user], [items[item] for item in sequence]) for user, sequence in rows(Path(versioned) / filename)
            ]
        except KeyError as error:
            raise ValueError(f'evaluation mapping mismatch: {filename}') from error
        if before != after or len(dict(before)) != len(before):
            raise ValueError(f'evaluation cohort/history/target mismatch: {filename}')
    return True


def legacy_catalogue_size(directory):
    maximum = 0
    for filename in FILES:
        if filename == 'train_sasrec.txt':
            continue
        for _, sequence in rows(Path(directory) / filename):
            if not filename.endswith('_target.txt'):
                sequence = sequence[-20:]
            maximum = max(maximum, max(sequence, default=0))
    return maximum


def prepare_study(
    sequences, raw_path, legacy_reference, output, repo_root, template_path, steps=250000, validation_every=256
):
    if type(steps) is not int or steps < 1 or type(validation_every) is not int or not 1 <= validation_every <= steps:
        raise ValueError('steps and validation_every must be positive, with at least one validation')
    output, repo_root, legacy_reference = (
        Path(output).resolve(),
        Path(repo_root).resolve(),
        Path(legacy_reference).resolve(),
    )
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output.name):
        raise ValueError('study directory name must be a safe experiment prefix')
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / '.preparation-incomplete'
    incomplete.touch(exist_ok=False)
    data_root = output / 'data'
    data = {}
    for label, protocol in (('legacy', 'legacy-v1'), ('v2', 'windowed-v2')):
        directory = data_root / label
        count = write_files(sequences, *split_users(sequences), directory, protocol=protocol)
        data[label] = {'root': str(directory), 'train_targets': count, 'protocol': protocol}
    legacy = data_root / 'legacy'
    for filename in FILES:
        if sha256(legacy / filename) != sha256(legacy_reference / filename):
            raise ValueError(f'legacy reference differs: {filename}; no study was frozen')
    verify_evaluation(legacy, data_root / 'v2')
    manifest = json.loads((data_root / 'v2/preprocessing.json').read_text())
    legacy_items = legacy_catalogue_size(legacy)
    if set(map(int, manifest['item_id_mapping'])) != set(range(1, legacy_items + 1)):
        raise ValueError('candidate catalogues differ; this study requires the same positive dense original item IDs')
    for label, record in data.items():
        directory = Path(record['root'])
        num_items = legacy_items if label == 'legacy' else manifest['num_items']
        subprocess.run(
            [
                sys.executable,
                str(repo_root / 'scripts/generate_mclsr_role_counts.py'),
                '--input',
                str(directory / 'train_mclsr.txt'),
                '--num_items',
                str(num_items),
                '--target_output',
                str(directory / 'item_target_counts.pkl'),
                '--context_output',
                str(directory / 'item_context_counts.pkl'),
            ],
            check=True,
        )
        record['num_items'] = num_items
    subprocess.run(
        [
            sys.executable,
            str(repo_root / 'scripts/generate_item_counts.py'),
            '--input',
            str(legacy / 'train_sasrec.txt'),
            '--num_items',
            str(legacy_items),
            '--output',
            str(legacy / 'item_counts.pkl'),
        ],
        check=True,
    )
    proxy_reference_match = None
    if (legacy_reference / 'item_counts.pkl').exists():
        with (
            (legacy_reference / 'item_counts.pkl').open('rb') as source,
            (legacy / 'item_counts.pkl').open('rb') as generated,
        ):
            proxy_reference_match = np.array_equal(pickle.load(source), pickle.load(generated))
        if not proxy_reference_match:
            raise ValueError('legacy proxy counts differ from historical reference')
    for record in data.values():
        directory = Path(record['root'])
        record['files_sha256'] = {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
    template = json.loads(Path(template_path).read_text())
    if template['dataset']['dataset']['type'] != 'mclsr' or template['model']['type'] != 'mclsr':
        raise ValueError('this study requires an MCLSR template')
    if template['dataloader']['train']['batch_size'] > min(record['train_targets'] for record in data.values()):
        raise ValueError('training dataset is smaller than one batch')
    configs = {}
    config_dir = output / 'configs'
    config_dir.mkdir()
    study_path = output / 'study.json'
    for arm in ARMS:
        label = 'v2' if arm.startswith('v2') else 'legacy'
        config = copy.deepcopy(template)
        config.update(
            experiment_name=f'{output.name}_{arm}',
            train_steps_num=steps,
            epochs_threshold=steps,
            best_metric='validation/ndcg@20',
            path_to_study_manifest=str(study_path),
        )
        config.pop('train_epochs_num', None)
        config['dataset'].update(graph_dir_path=data[label]['root'], use_train_data_only=True)
        config['dataset']['dataset'].update(path_to_data_dir=str(data_root), name=label, max_sequence_length=20)
        config['dataset']['dataset'].pop('eval_history_max_len', None)
        config['dataset']['dataset'].pop('max_history_length', None)
        if label == 'v2':
            config['dataset']['dataset']['max_history_length'] = 20
        retrieval = config['loss']['losses'][0]
        target_counts = arm != 'legacy_proxy_l10'
        retrieval.update(
            path_to_item_counts=str(
                Path(data[label]['root']) / ('item_target_counts.pkl' if target_counts else 'item_counts.pkl')
            ),
            logq_lambda=0.0 if arm.endswith('l00') else 1.0,
        )
        if target_counts:
            retrieval.update(expected_counts_role='target', expected_counts_max_len=21 if label == 'v2' else 20)
        else:
            retrieval.pop('expected_counts_role', None)
            retrieval.pop('expected_counts_max_len', None)
        for callback in config['callback']['callbacks']:
            if callback['type'] in ('validation', 'eval'):
                callback.update(
                    on_step=validation_every,
                    metrics={
                        'ndcg@20': {'type': 'mclsr-ndcg', 'k': 20},
                        'recall@1000': {'type': 'mclsr-recall', 'k': 1000},
                    },
                )
        config_path = config_dir / f'{arm}.json'
        write_json(config_path, config)
        configs[arm] = {'path': str(config_path), 'sha256': sha256(config_path)}
    runs = [
        {'index': index, 'arm': arm, 'seed': seed}
        for index, (seed, arm) in enumerate((seed, arm) for seed in (1, 2, 3) for arm in ARMS)
    ]
    write_json(
        study_path,
        {
            'schema_version': 1,
            'name': output.name,
            'root': str(output),
            'repo_root': str(repo_root),
            'raw_source': {'path': str(Path(raw_path).resolve()), 'sha256': sha256(raw_path)},
            'template': {'path': str(Path(template_path).resolve()), 'sha256': sha256(template_path)},
            'source_sha256': source_hashes(repo_root),
            'data': data,
            'configs': configs,
            'runs': runs,
            'steps': steps,
            'workers': 2,
            'validation_every': validation_every,
            'selection': ['validation/ndcg@20'],
            'evaluation_equivalent': True,
            'candidate_catalogue_equivalent': True,
            'evaluation_history_capacity': 20,
            'legacy_reference_exact': True,
            'proxy_reference_match': bool(proxy_reference_match) if proxy_reference_match is not None else None,
            'scope': 'Data/proposal intervention on a common executable, not historical executable replication',
            'proposal_scope': 'Marginal ladder target frequencies; not exact anchor-conditional masked sampling probabilities',
            'contrasts': {
                'data_without_correction': ['v2_l00', 'legacy_l00'],
                'proposal_only': ['legacy_target_l10', 'legacy_proxy_l10'],
                'data_at_target_proposal': ['v2_target_l10', 'legacy_target_l10'],
                'legacy_logq': ['legacy_target_l10', 'legacy_l00'],
                'v2_logq': ['v2_target_l10', 'v2_l00'],
                'combined_data_and_proposal': ['v2_target_l10', 'legacy_proxy_l10'],
            },
        },
    )
    incomplete.unlink()
    return study_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', type=Path)
    source.add_argument('--seq', type=Path)
    parser.add_argument('--legacy-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--validation-every', type=int, default=256)
    args = parser.parse_args()
    sequences = sequences_from_csv(args.csv) if args.csv else sequences_from_seq_file(args.seq)
    path = prepare_study(
        sequences,
        args.csv or args.seq,
        args.legacy_reference,
        args.output,
        Path(__file__).resolve().parents[1],
        args.template,
        args.steps,
        args.validation_every,
    )
    print(f'STUDY_READY {path}', flush=True)


if __name__ == '__main__':
    main()
