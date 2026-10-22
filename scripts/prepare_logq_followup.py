"""Prepare two additional seed pairs from an already completed auxiliary-logQ screen."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

ARMS = ('masked_dq1_auxq0', 'masked_dq1_user_feature_q1')
SEEDS = (2, 3)


def matched_configs(parent, screen):
    configs = {}
    for arm in ARMS:
        record = parent['configs'][arm]
        if screen.sha256(record['path']) != record['sha256']:
            raise ValueError('parent configuration changed')
        configs[arm] = json.loads(Path(record['path']).read_text())
    control = configs[ARMS[0]]
    losses = control['loss']['losses']
    expected_prefixes = {
        'downstream_loss',
        'contrastive_interest_loss',
        'contrastive_user_feature_loss',
        'contrastive_item_feature_loss',
    }
    if len(losses) != 4 or {loss['output_prefix'] for loss in losses} != expected_prefixes:
        raise ValueError('matched control must contain all four objectives')
    for loss in losses:
        if loss['output_prefix'] == 'downstream_loss':
            if loss.get('logq_lambda') != 1.0:
                raise ValueError('matched control requires downstream Q=1')
        elif loss.get('type') != 'fps_logq' or loss.get('logq_lambda') != 0.0 or not loss.get('mask_false_negatives'):
            raise ValueError('matched control requires masked auxiliaries with Q=0')
    expected = copy.deepcopy(control)
    expected['experiment_name'] = configs[ARMS[1]]['experiment_name']
    next(loss for loss in expected['loss']['losses'] if loss['output_prefix'] == 'contrastive_user_feature_loss')[
        'logq_lambda'
    ] = 1.0
    if expected != configs[ARMS[1]]:
        raise ValueError('parent pair is not matched: only user-feature Q may differ')
    return configs


def prepare_followup(parent_path, parent_sha256, output, screen):
    parent_path, output = Path(parent_path).resolve(), Path(output).resolve()
    if screen.sha256(parent_path) != parent_sha256:
        raise ValueError('unexpected parent manifest hash')
    parent = screen.validate_screen(parent_path)
    required = {
        'seed': 1,
        'steps': 65536,
        'validation_every': 256,
        'selection': 'validation/ndcg@20',
        'test_evaluation': False,
    }
    if any(parent.get(key) != value for key, value in required.items()):
        raise ValueError('unexpected parent protocol, budget, seed, or test evaluation')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output.name):
        raise ValueError('output name must be a safe experiment prefix')
    if output.exists():
        raise FileExistsError(output)
    configs = matched_configs(parent, screen)
    result_hashes = {}
    for arm in ARMS:
        screen.validate_result(parent, arm, parent_path)
        result = Path(parent['root']) / 'runs' / arm / 'result.json'
        result_hashes[str(result)] = screen.sha256(result)
    plans = []
    for seed in SEEDS:
        for arm in ARMS:
            config = copy.deepcopy(configs[arm])
            config['experiment_name'] = f'{output.name}_seed{seed}_{arm}'
            for directory in ('checkpoints', 'tensorboard_logs'):
                if any((Path(parent['repo_root']) / directory).glob(config['experiment_name'] + '*')):
                    raise FileExistsError(f'experiment namespace exists: {config["experiment_name"]}')
            plans.append((seed, arm, config))
    if screen.sha256(parent_path) != parent_sha256:
        raise ValueError('parent manifest changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    index = {
        'kind': 'uc_logq_seed_followup',
        'status': 'prepared_not_started',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'parent': {'path': str(parent_path), 'sha256': parent_sha256, 'results_sha256': result_hashes},
        'seeds': list(SEEDS),
        'arms': list(ARMS),
        'queues': [],
        'test_evaluation': False,
        'comparison': 'Paired UC-Q minus matched-control deltas per seed; do not select the best seed.',
        'execution': 'Manual start only, on an idle Slurm GPU allocation; no scheduler submission or automatic retry.',
        'scope': 'No architecture, paper-replication, loss-weight, data, or environment changes.',
    }
    for seed in SEEDS:
        root = output / f'seed_{seed}'
        (root / 'configs').mkdir(parents=True)
        records = {}
        for planned_seed, arm, config in plans:
            if planned_seed != seed:
                continue
            path = root / 'configs' / f'{arm}.json'
            screen.write_json(path, config)
            records[arm] = {'path': str(path), 'sha256': screen.sha256(path)}
        manifest = copy.deepcopy(parent)
        manifest.update(
            root=str(root),
            seed=seed,
            arms=list(ARMS),
            configs=records,
            followup_of=index['parent'],
            followup_scope=index['comparison'],
        )
        manifest['tool_sha256'][str(Path(__file__).resolve())] = screen.sha256(__file__)
        manifest['artifacts_sha256'].update(result_hashes)
        manifest['artifacts_sha256'][str(parent_path)] = parent_sha256
        path = root / 'screen.json'
        screen.write_json(path, manifest)
        screen.validate_screen(path)
        index['queues'].append(
            {
                'seed': seed,
                'manifest': str(path),
                'sha256': screen.sha256(path),
                'command': [
                    sys.executable,
                    '-u',
                    str(Path(screen.__file__).resolve()),
                    'queue',
                    '--manifest',
                    str(path),
                ],
            }
        )
    screen.validate_screen(parent_path)
    if screen.sha256(parent_path) != parent_sha256:
        raise ValueError('parent manifest changed during preparation')
    index_path = output / 'followup.json'
    screen.write_json(index_path, index)
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--screen-tool', type=Path, required=True)
    args = parser.parse_args()
    tool_path = args.screen_tool.resolve()
    sys.path.insert(0, str(tool_path.parent))
    spec = importlib.util.spec_from_file_location('followup_screen_tool', tool_path)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    print(prepare_followup(args.parent, args.parent_sha256, args.output_dir, screen))


if __name__ == '__main__':
    main()
