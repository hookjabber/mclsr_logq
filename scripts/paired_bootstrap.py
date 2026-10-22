#!/usr/bin/env python
"""Paired, per-user uncertainty conditional on the trained seeds.

For every ``arm:baseline`` pair and every seed, the validation-selected checkpoints of both arms
are evaluated on the same test users, aligned by unique user IDs across arms and seeds. The
mean difference gets a 95% bootstrap interval resampling users, keeping each user's trained
seeds together, and a two-sided user-level sign-flip p-value. Neither estimates uncertainty
over new training runs or establishes per-seed significance or equivalence. Per-seed means
and their descriptive spread are reported separately. Existing output files are never replaced.
Config and checkpoint hashes are mandatory. When a provenance sidecar is present, its recorded
dataset inputs are verified too; legacy reports without it cannot certify the evaluation data.

    python scripts/paired_bootstrap.py --prefix beauty --config-dir configs/train/beauty \
        --seeds 1 2 3 --pairs 02_logq_downstream:01_orig 03_graph:03_graph_l00 \
        --output results/stats/beauty_paired_bootstrap.md
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
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
    user_ids = []
    user_ids_key = f'{config["model"].get("user_prefix", "user")}.ids'
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch[pred_prefix] = model(batch)
            batch = {k: v.cpu() for k, v in batch.items()}
            if user_ids_key not in batch:
                raise ValueError(f'Missing user IDs in evaluation batch: {user_ids_key}')
            batch_user_ids = batch[user_ids_key].reshape(-1).numpy()
            if batch_user_ids.size != batch[pred_prefix].shape[0]:
                raise ValueError('Evaluation requires one user ID per prediction row')
            user_ids.extend(batch_user_ids.tolist())
            for name, metric in metrics.items():
                batch_values = metric(inputs=batch, pred_prefix=pred_prefix, labels_prefix=labels_prefix)
                if len(batch_values) != batch_user_ids.size:
                    raise ValueError(f'{name} must return one value per user')
                values[name].extend(batch_values)
    return {
        'user_ids': np.asarray(user_ids),
        **{name: np.asarray(metric_values, dtype=np.float64) for name, metric_values in values.items()},
    }


def aligned_user_metrics(result, expected_user_ids=None):
    """Validate a cohort and sort every metric by its actual user IDs, never by row position."""
    if 'user_ids' not in result:
        raise ValueError('Per-user metrics must include user_ids')
    user_ids = np.asarray(result['user_ids'])
    if user_ids.ndim != 1 or user_ids.size == 0 or user_ids.dtype.kind not in 'iu':
        raise ValueError('user_ids must be a nonempty one-dimensional integer array')
    if np.unique(user_ids).size != user_ids.size:
        raise ValueError('Each evaluation must contain unique user IDs')
    order = np.argsort(user_ids)
    sorted_user_ids = user_ids[order]
    if expected_user_ids is not None and not np.array_equal(sorted_user_ids, expected_user_ids):
        raise ValueError('All arms and seeds must score exactly the same user IDs')
    aligned = {'user_ids': sorted_user_ids}
    for name in METRICS:
        if name not in result:
            raise ValueError(f'Missing per-user metric: {name}')
        values = np.asarray(result[name], dtype=np.float64)
        if values.shape != user_ids.shape:
            raise ValueError(f'{name} must contain exactly one value per user ID')
        if not np.isfinite(values).all():
            raise ValueError(f'{name} must contain finite per-user values')
        aligned[name] = values[order]
    return aligned


def checked_sha256(path, expected, label, payload=None):
    if not isinstance(expected, str) or len(expected) != 64 or set(expected.lower()) - set('0123456789abcdef'):
        raise ValueError(f'{label} requires a valid SHA-256: {path}')
    if payload is None:
        with open(path, 'rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
    else:
        actual = hashlib.sha256(payload).hexdigest()
    if actual != expected.lower():
        raise ValueError(f'{label} SHA-256 mismatch: {path}; expected {expected}, got {actual}')
    return actual


def verified_json(path, expected, label):
    payload = Path(path).read_bytes()
    checked_sha256(path, expected, label, payload)
    return json.loads(payload)


def dataset_roots(node):
    if isinstance(node, dict):
        if 'path_to_data_dir' in node and 'name' in node:
            yield Path(node['path_to_data_dir']) / node['name']
        for child in node.values():
            yield from dataset_roots(child)
    elif isinstance(node, list):
        for child in node:
            yield from dataset_roots(child)


def verify_provenance(report, report_path, config):
    try:
        reference = report['provenance']
        path = Path(reference['path'])
        if not path.is_file():
            path = report_path.parent / path.name
        manifest = verified_json(path, reference.get('sha256'), 'provenance sidecar')
        if manifest['schema_version'] != 1 or manifest['seed'] != report['seed']:
            raise ValueError('Unsupported provenance schema or mismatched seed')
        if manifest['params']['sha256'] != report['params_sha256']:
            raise ValueError('provenance params SHA-256 does not match the report')
        if manifest['evaluation_config'] != get_inference_config(config, 'eval'):
            raise ValueError('provenance evaluation_config does not match the supplied config')
        working_directory = Path(manifest['working_directory'])
        if not working_directory.is_absolute():
            raise ValueError('provenance working_directory must be absolute')
        locations = {(working_directory / root).resolve(): root.resolve() for root in dataset_roots(config['dataset'])}
        records = {Path(entry['root']).resolve(): entry['inputs'] for entry in manifest['datasets']}
        if not locations or set(locations) != set(records) or len(records) != len(manifest['datasets']):
            raise ValueError('provenance datasets do not match the evaluation input locations')
        for recorded_root, actual_root in locations.items():
            inputs = records[recorded_root]
            if not {'test_history.txt', 'test_target.txt'} <= inputs.keys():
                raise ValueError('provenance must include evaluation history and target input hashes')
            for name, record in inputs.items():
                if Path(name).name != name or Path(record['path']).resolve() != recorded_root / name:
                    raise ValueError(f'provenance input path does not match its dataset: {name}')
                checked_sha256(actual_root / name, record.get('sha256'), f'provenance input {name}')
        return reference['sha256']
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Invalid provenance structure in {report_path}: {error}') from error


def load_arm(prefix, arm, seed, config_dir, confirm_dir, cache):
    report_path = (Path(confirm_dir) / f'{prefix}_{arm}_seed{seed}.json').resolve()
    config_path = (Path(config_dir) / f'{arm}.json').resolve()
    report = json.loads(report_path.read_bytes())
    config = verified_json(config_path, report.get('params_sha256'), 'config')
    selected = report['results']['validation/ndcg@20']
    if report.get('seed') != seed or selected.get('selection_metric') != 'validation/ndcg@20':
        raise ValueError('Report seed or checkpoint selection does not match the requested arm')
    checkpoint = Path(selected['checkpoint']).resolve()
    checkpoint_hash = checked_sha256(checkpoint, selected.get('checkpoint_sha256'), 'checkpoint')
    provenance_hash = verify_provenance(report, report_path, config) if 'provenance' in report else None
    key = (
        str(report_path),
        str(config_path),
        report['params_sha256'],
        str(checkpoint),
        checkpoint_hash,
        provenance_hash,
    )
    if key not in cache:
        if provenance_hash is not None:
            _DATASETS.pop(json.dumps(config['dataset'], sort_keys=True), None)
        cache[key] = per_user_metrics(copy.deepcopy(config), str(checkpoint))
    return cache[key]


def paired_stats(differences, rng, draws=10000):
    """User-conditional statistics for a (users, fixed trained seeds) difference matrix.

    The interval excludes training-run uncertainty. The sign-flip test assumes independent
    users with sign-exchangeable differences under its null; it is not a per-seed test.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or 0 in differences.shape or not np.isfinite(differences).all():
        raise ValueError('differences must be a nonempty finite (users, seeds) matrix')
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws <= 0:
        raise ValueError('draws must be a positive integer')
    num_users = differences.shape[0]
    row_means = differences.mean(axis=1)
    observed = row_means.mean()
    indices = rng.integers(0, num_users, size=(draws, num_users))
    bootstrap_means = row_means[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(draws, num_users))
    permuted_means = (signs * row_means[None, :]).mean(axis=1)
    p_value = (np.sum(np.abs(permuted_means) >= abs(observed)) + 1) / (draws + 1)
    return observed, lower, upper, p_value


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

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Training seeds must be unique')
    if args.draws <= 0:
        raise ValueError('draws must be a positive integer')
    json_output = os.path.splitext(args.output)[0] + '.json'
    if args.output == json_output:
        raise ValueError('The output path must differ from its .json sidecar')
    for output_path in (args.output, json_output):
        if os.path.lexists(output_path):
            raise FileExistsError(f'Output already exists; choose a new path to preserve it: {output_path}')

    rng = np.random.default_rng(0)
    cache = {}
    description = (
        'Inference is conditional on the trained seeds listed below; it does not estimate '
        'uncertainty over new training runs. Both metrics use checkpoints selected by validation/ndcg@20 '
        '(including recall, not a recall-selected checkpoint). Users are aligned by unique IDs across '
        "all arms and seeds, then resampled independently with all of a user's seeds kept together. "
        f'The 95% CI uses a user bootstrap ({args.draws} draws); the two-sided sign-flip test '
        f'uses {args.draws} draws and assumes user-level sign exchangeability under the null. '
        'Per-seed differences and their sample SD are descriptive, not per-seed significance. '
        'A nonsignificant result is not evidence of equivalence. P-values are unadjusted for '
        'multiple comparisons; these tests alone do not establish confirmatory evidence.'
    )
    print(description, flush=True)
    lines = [
        '| arm − baseline | metric | per-seed Δ (descriptive) | seed SD (descriptive) | mean Δ | 95% CI (conditional user bootstrap) | p (conditional sign-flip, unadjusted) | users × trained seeds |',
        '|---|---|---|---|---|---|---|---|',
    ]
    out = {}
    reference_user_ids = None
    for pair in args.pairs:
        arm, base = pair.split(':')
        per_metric = {name: [] for name in METRICS}
        for seed in args.seeds:
            arm_values = aligned_user_metrics(
                load_arm(args.prefix, arm, seed, args.config_dir, args.confirm_dir, cache), reference_user_ids
            )
            reference_user_ids = arm_values['user_ids']
            base_values = aligned_user_metrics(
                load_arm(args.prefix, base, seed, args.config_dir, args.confirm_dir, cache), reference_user_ids
            )
            for name in METRICS:
                per_metric[name].append(arm_values[name] - base_values[name])
        for name in METRICS:
            differences = np.stack(per_metric[name], axis=1)
            observed, lower, upper, p_value = paired_stats(differences, rng, args.draws)
            seed_means = differences.mean(axis=0)
            seed_sd = float(seed_means.std(ddof=1)) if seed_means.size > 1 else None
            seed_summary = ', '.join(f'seed {seed}: {mean:+.4f}' for seed, mean in zip(args.seeds, seed_means))
            seed_sd_summary = f'{seed_sd:.4f}' if seed_sd is not None else 'n/a (one seed)'
            lines.append(
                f'| {arm} − {base} | {name} | {seed_summary} | {seed_sd_summary} | {observed:+.4f} | '
                f'[{lower:+.4f}, {upper:+.4f}] | {p_value:.4f} | {differences.shape[0]} × {differences.shape[1]} |'
            )
            out[f'{arm}:{base}:{name}'] = {
                'per_seed': seed_means.tolist(),
                'per_seed_sd': seed_sd,
                'mean': observed,
                'ci95': [lower, upper],
                'p_sign_flip': p_value,
                'users': int(differences.shape[0]),
                'seeds': args.seeds,
                'user_ids': reference_user_ids.tolist(),
                'per_user_differences': differences.tolist(),
                'inference_scope': 'conditional_on_trained_seeds',
                'resampling_unit': 'user',
                'training_seed_uncertainty': 'not_estimated',
                'per_seed_inference': 'not_performed',
                'equivalence_test': 'not_performed',
                'multiple_testing_adjustment': 'none',
                'checkpoint_selection_metric': 'validation/ndcg@20',
                'draws': args.draws,
                'resampling_random_seed': 0,
            }
            print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'x') as report_file:
        report_file.write(f'# Paired per-user comparisons — {args.prefix}\n\n')
        report_file.write(description + '\n\n')
        report_file.write('\n'.join(lines) + '\n')
    with open(json_output, 'x') as report_file:
        json.dump(out, report_file, indent=1, allow_nan=False)
    print('written', args.output)


if __name__ == '__main__':
    main()
