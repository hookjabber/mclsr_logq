"""Freeze and run a validation-only component/logQ screen beside an existing study."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

COMPONENTS = ('interest', 'user_feature', 'item_feature')
SELECTION = 'validation/ndcg@20'


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write('\n')


def build_configs(base, name, user_counts, steps, validation_every):
    if type(steps) is not int or type(validation_every) is not int or not 1 <= validation_every <= steps:
        raise ValueError('invalid update budget or validation interval')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('screen name must be a safe experiment prefix')
    baseline = copy.deepcopy(base)
    baseline.update(train_steps_num=steps, epochs_threshold=steps, best_metric=SELECTION)
    baseline.pop('train_epochs_num', None)
    baseline.pop('path_to_study_manifest', None)
    by_prefix = {loss['output_prefix']: loss for loss in baseline['loss']['losses']}
    expected = {'downstream_loss'} | {f'contrastive_{component}_loss' for component in COMPONENTS}
    if set(by_prefix) != expected or any(
        by_prefix[f'contrastive_{component}_loss']['type'] != 'fps' for component in COMPONENTS
    ):
        raise ValueError('screen requires the four-loss uncorrected-contrastive MCLSR baseline')
    for key, value in {
        'variant': 'negatives_only',
        'temperature': 1.0,
        'normalize_embeddings': False,
        'leave_own_out': False,
        'correct_positive': False,
        'mixed_uniform_negatives': 0,
    }.items():
        by_prefix['downstream_loss'].setdefault(key, value)
    for callback in baseline['callback']['callbacks']:
        if callback['type'] in ('validation', 'eval'):
            callback.update(
                on_step=validation_every,
                metrics={
                    'ndcg@20': {'type': 'mclsr-ndcg', 'k': 20},
                    'recall@1000': {'type': 'mclsr-recall', 'k': 1000},
                },
            )
    configs = {}
    objectives = {
        'full': (),
        **{f'no_{component}': (component,) for component in COMPONENTS},
        'downstream_only': COMPONENTS,
    }
    for objective, removed in objectives.items():
        for correction in (0, 1):
            arm = f'{objective}_dq{correction}'
            config = copy.deepcopy(baseline)
            disabled = {f'contrastive_{component}_loss' for component in removed}
            config['loss']['losses'] = [
                loss for loss in config['loss']['losses'] if loss['output_prefix'] not in disabled
            ]
            next(loss for loss in config['loss']['losses'] if loss['output_prefix'] == 'downstream_loss')[
                'logq_lambda'
            ] = float(correction)
            config['callback']['callbacks'] = [
                callback for callback in config['callback']['callbacks'] if callback.get('loss_prefix') not in disabled
            ]
            config['experiment_name'] = f'{name}_{arm}'
            configs[arm] = config
    control = copy.deepcopy(configs['full_dq1'])
    for loss in control['loss']['losses']:
        if loss['output_prefix'] == 'downstream_loss':
            continue
        item_loss = loss['output_prefix'] == 'contrastive_item_feature_loss'
        loss.update(
            type='fps_logq',
            logq_lambda=0.0,
            mask_false_negatives=True,
            leave_own_out=False,
            center_log_q=False,
            ids_prefix='item_ids' if item_loss else 'user_ids',
            path_to_counts=str(Path(base['dataset']['graph_dir_path']) / 'item_context_counts.pkl')
            if item_loss
            else str(user_counts),
            expected_counts_role='context-inclusion' if item_loss else 'user-row',
            expected_counts_max_len=21,
            logq_probability_mode='inbatch_negative' if item_loss else 'sample',
        )
        if item_loss:
            loss['num_draws_value'] = base['dataloader']['train']['batch_size']
    variants = {'auxq0': (), **{f'{component}_q1': (component,) for component in COMPONENTS}, 'all_q1': COMPONENTS}
    for suffix, corrected in variants.items():
        arm = f'masked_dq1_{suffix}'
        config = copy.deepcopy(control)
        config['experiment_name'] = f'{name}_{arm}'
        for loss in config['loss']['losses']:
            if loss['output_prefix'] in {f'contrastive_{component}_loss' for component in corrected}:
                loss['logq_lambda'] = 1.0
        configs[arm] = config
    return configs


def write_user_counts(source, output, num_users, max_len):
    import numpy as np

    counts = np.zeros(num_users + 2, dtype=np.int64)
    with Path(source).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            user = int(fields[0])
            if not 1 <= user <= num_users or len(fields) < 3:
                raise ValueError('invalid ladder row for user proposal')
            counts[user] += 1
    if not counts.sum():
        raise ValueError('empty ladder')
    with Path(output).open('xb') as destination:
        pickle.dump(
            {
                'counts': counts,
                'denominator': int(counts.sum()),
                'num_lines': int(counts.sum()),
                'role': 'user-row',
                'max_len': max_len,
                'source': str(Path(source).resolve()),
                'source_sha256': sha256(source),
            },
            destination,
        )


def study_modules(study_path):
    preliminary = json.loads(Path(study_path).read_text())
    root = Path(preliminary['repo_root'])
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    from run_protocol_study import validate_study

    return validate_study(study_path)


def validate_source_overlay(study, source_root=None):
    from prepare_protocol_study import source_hashes

    root = Path(source_root or study['repo_root']).resolve()
    hashes = source_hashes(root)
    baseline = study['source_sha256']
    changed = sorted(name for name in set(hashes) | set(baseline) if hashes.get(name) != baseline.get(name))
    if set(hashes) != set(baseline) or not set(changed) <= {'src/irec/loss/counts.py', 'src/irec/loss/contrastive.py'}:
        raise ValueError('source snapshot differs beyond the two audited loss repairs')
    return root, hashes, changed


def prepare_screen(study_path, output, steps=65536, seed=1, launcher=None, source_root=None):
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    study = study_modules(study_path)
    execution_root, source_hashes, changed = validate_source_overlay(study, source_root)
    if not 1 <= steps <= study['steps']:
        raise ValueError('screen cannot exceed the original update budget')
    base = json.loads(Path(study['configs']['v2_l00']['path']).read_text())
    dataset = base['dataset']['dataset']
    if dataset.get('max_history_length') != 20 or dataset['max_sequence_length'] != 20:
        raise ValueError('screen requires the frozen v2 history capacity of 20')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configs').mkdir()
    data = Path(study['data']['v2']['root'])
    metadata = json.loads((data / 'preprocessing.json').read_text())
    user_counts = output / 'user_row_counts.pkl'
    write_user_counts(data / 'train_mclsr.txt', user_counts, metadata['num_users'], 21)
    configs = build_configs(base, output.name, user_counts, steps, study['validation_every'])
    records = {}
    for arm, config in configs.items():
        config_path = output / 'configs' / f'{arm}.json'
        write_json(config_path, config)
        records[arm] = {'path': str(config_path), 'sha256': sha256(config_path)}
    import diagnose_training

    tools = [Path(__file__).resolve(), Path(diagnose_training.__file__).resolve()]
    if launcher:
        tools.append(Path(launcher).resolve())
    manifest = {
        'kind': 'exploratory_validation_only_component_screen',
        'root': str(output),
        'repo_root': str(execution_root),
        'source_sha256': source_hashes,
        'source_changes_from_frozen_study': changed,
        'study': str(study_path),
        'study_sha256': sha256(study_path),
        'seed': seed,
        'steps': steps,
        'validation_every': study['validation_every'],
        'selection': SELECTION,
        'configs': records,
        'arms': list(configs),
        'tool_sha256': {str(path): sha256(path) for path in tools},
        'artifacts_sha256': {str(user_counts): sha256(user_counts)},
        'launcher': str(Path(launcher).resolve()) if launcher else None,
        'test_evaluation': False,
        'objective_scope': 'Remove loss terms and metric callbacks, not model graph branches or architecture.',
        'proposal_scope': {
            'downstream': 'Frozen target-row marginal; unchanged downstream correction mode.',
            'interest_user': 'User row frequency, sample mode; multiplicities retained; leave-own-out off.',
            'item': 'Approximate marginal IID union inclusion 1-(1-p_context)^batch_rows, not unique-item count. Not exact without-replacement or anchor-conditional inclusion.',
            'margin': 'Positive is uncorrected for auxiliary Q; uncentered Q also changes the positive/negative margin.',
        },
        'failure_policy': 'No retries. Preserve failures. Continue after isolated native SIGSEGV/SIGBUS only; stop on two consecutive or three total native failures, any other error, or changed inputs.',
    }
    path = output / 'screen.json'
    write_json(path, manifest)
    return path


def validate_screen(path):
    manifest = json.loads(Path(path).read_text())
    if sha256(manifest['study']) != manifest['study_sha256']:
        raise ValueError('frozen study manifest changed')
    study = study_modules(manifest['study'])
    execution_root, hashes, changed = validate_source_overlay(study, manifest['repo_root'])
    if hashes != manifest['source_sha256'] or changed != manifest['source_changes_from_frozen_study']:
        raise ValueError('screen execution source changed')
    records = {
        **manifest['tool_sha256'],
        **manifest['artifacts_sha256'],
        **{record['path']: record['sha256'] for record in manifest['configs'].values()},
    }
    for filename, expected in records.items():
        if sha256(filename) != expected:
            raise ValueError(f'screen input changed: {filename}')
    sys.path[:0] = [str(execution_root / 'src'), str(execution_root / 'scripts')]
    return manifest


def collect_validation(runner, arguments):
    from diagnose_training import run_training_only

    original = runner.train
    best_validation = {}
    last_validation = {}

    def captured_train(**kwargs):
        callback = kwargs['callback']

        def capture(batch, step):
            callback(batch, step)
            metrics = {name: float(value) for name, value in batch.items() if name.startswith('validation/')}
            if metrics:
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError('nonfinite validation metric')
                last_validation.update(step=step, metrics=metrics)

        kwargs['callback'] = capture
        best = original(**kwargs)
        if set(best) != {SELECTION}:
            raise ValueError('missing expected validation selection')
        for name, record in best.items():
            if not math.isfinite(record['value']):
                raise ValueError('nonfinite selected validation metric')
            best_validation[name] = {key: value for key, value in record.items() if key != 'state'}
        return best

    runner.train = captured_train
    try:
        result = run_training_only(runner, arguments)
        return dict(result, best_validation=best_validation, last_validation=last_validation)
    finally:
        runner.train = original


def run_arm(path, arm):
    digest = sha256(path)
    manifest = validate_screen(path)
    config = manifest['configs'][arm]
    output = Path(manifest['root']) / 'runs' / arm
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        'screen_confirmatory', Path(manifest['repo_root']) / 'scripts/train_confirmatory.py'
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    os.chdir(manifest['repo_root'])
    unused = output / 'unused_test_report.json'
    result = collect_validation(
        runner,
        [
            '--params',
            config['path'],
            '--seed',
            str(manifest['seed']),
            '--max-steps',
            str(manifest['steps']),
            '--select',
            SELECTION,
            '--output',
            str(unused),
        ],
    )
    if unused.exists() or sha256(path) != digest:
        raise ValueError('test report exists or screen manifest changed')
    validate_screen(path)
    provenance = Path(str(unused) + '.provenance')
    record = json.loads(provenance.read_text())
    checkpoint = Path(manifest['repo_root']) / 'checkpoints' / f'{record["experiment"]}_best_validation_ndcg_at_20.pth'
    result.update(
        arm=arm,
        seed=manifest['seed'],
        screen_sha256=digest,
        params_sha256=config['sha256'],
        provenance={'path': str(provenance), 'sha256': sha256(provenance)},
        checkpoint={'path': str(checkpoint), 'sha256': sha256(checkpoint)},
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output / 'result.json', result)
    print('VALIDATION_ONLY_FINISHED', json.dumps(result), flush=True)


def validate_result(manifest, arm, path):
    result_path = Path(manifest['root']) / 'runs' / arm / 'result.json'
    result = json.loads(result_path.read_text())
    expected = {
        'arm': arm,
        'seed': manifest['seed'],
        'screen_sha256': sha256(path),
        'params_sha256': manifest['configs'][arm]['sha256'],
        'optimizer_updates': manifest['steps'],
        'test_evaluated': False,
        'status': 'training_finished',
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('screen result identity or budget mismatch')
    for name in ('provenance', 'checkpoint'):
        if sha256(result[name]['path']) != result[name]['sha256']:
            raise ValueError(f'{name} hash mismatch')
    if not math.isfinite(result['best_validation'][SELECTION]['value']):
        raise ValueError('nonfinite validation result')
    return result


def run_queue(path):
    path = Path(path).resolve()
    digest = sha256(path)
    manifest = validate_screen(path)
    root = Path(manifest['root'])
    write_json(
        root / 'queue.reservation.json',
        {
            'pid': os.getpid(),
            'screen_sha256': digest,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        },
    )
    (root / 'logs').mkdir(exist_ok=True)
    failed, consecutive = [], 0
    attempted = []
    with (root / 'queue.events.jsonl').open('x', buffering=1) as events:
        for arm in manifest['arms']:
            if sha256(path) != digest:
                raise ValueError('screen manifest changed while queue was running')
            validate_screen(path)
            command = [
                sys.executable,
                '-u',
                str(Path(__file__).resolve()),
                'arm',
                '--manifest',
                str(path),
                '--arm',
                arm,
            ]
            if manifest.get('launcher'):
                command = ['bash', manifest['launcher'], *command]
            events.write(
                json.dumps({'arm': arm, 'status': 'running', 'time_utc': datetime.now(timezone.utc).isoformat()}) + '\n'
            )
            print(f'START {arm}', flush=True)
            with (root / 'logs' / f'{arm}.log').open('x') as log:
                outcome = subprocess.run(command, cwd=manifest['repo_root'], stdout=log, stderr=subprocess.STDOUT)
            attempted.append(arm)
            code, error = outcome.returncode, None
            if code == 0:
                try:
                    if sha256(path) != digest:
                        raise ValueError('screen manifest changed during training')
                    validate_screen(path)
                    validate_result(manifest, arm, path)
                except (ValueError, KeyError, TypeError, OSError) as failure:
                    code, error = 1, str(failure)
            status = 'complete' if code == 0 else 'failed'
            events.write(
                json.dumps(
                    {
                        'arm': arm,
                        'status': status,
                        'returncode': code,
                        'error': error,
                        'time_utc': datetime.now(timezone.utc).isoformat(),
                    }
                )
                + '\n'
            )
            print(f'{status.upper()} {arm} exit={code}', flush=True)
            consecutive = consecutive + 1 if code else 0
            if code:
                failed.append(arm)
                if code not in (-11, -7, 139, 135) or consecutive >= 2 or len(failed) >= 3:
                    break
    write_json(
        root / 'queue.summary.json',
        {
            'attempted': attempted,
            'failed': failed,
            'not_attempted': [arm for arm in manifest['arms'] if arm not in attempted],
            'test_evaluated': False,
        },
    )
    print('SCREEN_QUEUE_FINISHED' if not failed else 'SCREEN_QUEUE_FINISHED_WITH_FAILURES', flush=True)
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--study', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    prepare.add_argument('--steps', type=int, default=65536)
    prepare.add_argument('--seed', type=int, default=1)
    prepare.add_argument('--launcher', type=Path)
    prepare.add_argument('--source-root', type=Path)
    for command in ('queue', 'arm'):
        subparser = commands.add_parser(command)
        subparser.add_argument('--manifest', type=Path, required=True)
        if command == 'arm':
            subparser.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare_screen(args.study, args.output_dir, args.steps, args.seed, args.launcher, args.source_root))
    elif args.command == 'queue':
        raise SystemExit(run_queue(args.manifest))
    else:
        run_arm(args.manifest.resolve(), args.arm)


if __name__ == '__main__':
    main()
