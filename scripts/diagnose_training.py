"""Run bounded training diagnostics against a frozen study, never test evaluation."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


class TrainingFinished(Exception):
    pass


def run_training_only(runner, arguments):
    original_train, original_evaluate, original_argv = runner.train, runner.evaluate, sys.argv
    updates = 0

    def train_without_test(**kwargs):
        callback = kwargs['callback']

        def counted_callback(batch, step):
            nonlocal updates
            if step != updates:
                raise RuntimeError('diagnostic update counter is not contiguous')
            callback(batch, step)
            updates += 1

        kwargs['callback'] = counted_callback
        original_train(**kwargs)
        if updates != kwargs['step_cnt']:
            raise RuntimeError('diagnostic update budget was not completed')
        raise TrainingFinished

    def forbidden_evaluation(*args, **kwargs):
        raise RuntimeError('test evaluation is forbidden in a diagnostic run')

    runner.train, runner.evaluate = train_without_test, forbidden_evaluation
    sys.argv = ['train_confirmatory.py', *arguments]
    try:
        runner.main()
        raise RuntimeError('diagnostic runner returned without training')
    except TrainingFinished:
        return {'status': 'training_finished', 'optimizer_updates': updates, 'test_evaluated': False}
    finally:
        runner.train, runner.evaluate = original_train, original_evaluate
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('steps must be positive and label must be a safe experiment suffix')
    study_path = args.study.resolve()
    preliminary = json.loads(study_path.read_text())
    repo_root = Path(preliminary['repo_root']).resolve()
    sys.path.insert(0, str(repo_root / 'src'))
    sys.path.insert(0, str(repo_root / 'scripts'))
    from run_protocol_study import validate_study
    from prepare_protocol_study import sha256, write_json

    study = validate_study(study_path)
    if not any(run['arm'] == args.arm and run['seed'] == args.seed for run in study['runs']):
        parser.error('arm and seed must belong to the frozen study')
    if args.steps > study['steps']:
        parser.error('diagnostics cannot exceed the original training budget')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(json.loads(Path(study['configs'][args.arm]['path']).read_text()))
    config['experiment_name'] += f'_diagnostic_{args.label}'
    params = output_dir / 'params.json'
    write_json(params, config)
    manifest = {
        'kind': 'diagnostic_only_not_a_study_result',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'study': str(study_path),
        'study_sha256': sha256(study_path),
        'original_config_sha256': study['configs'][args.arm]['sha256'],
        'launcher_sha256': sha256(__file__),
        'params_sha256': sha256(params),
        'arm': args.arm,
        'seed': args.seed,
        'steps': args.steps,
        'test_evaluation': 'disabled; exit before returning from training to runner',
        'environment': {
            name: os.environ.get(name)
            for name in ('SLURM_JOB_ID', 'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'PYTHONFAULTHANDLER')
        },
    }
    write_json(output_dir / 'diagnostic.json', manifest)
    spec = importlib.util.spec_from_file_location(
        'diagnostic_confirmatory_runner', repo_root / 'scripts/train_confirmatory.py'
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    os.chdir(repo_root)
    try:
        result = run_training_only(
            runner,
            [
                '--params',
                str(params),
                '--seed',
                str(args.seed),
                '--max-steps',
                str(args.steps),
                '--select',
                *study['selection'],
                '--output',
                str(output_dir / 'unused_test_report.json'),
            ],
        )
        if sha256(study_path) != manifest['study_sha256'] or sha256(params) != manifest['params_sha256']:
            raise RuntimeError('diagnostic inputs changed during execution')
        validate_study(study_path)
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / 'result.json', result)
        print('DIAGNOSTIC_TRAINING_FINISHED_WITHOUT_TEST', json.dumps(result), flush=True)
    except Exception as error:
        write_json(output_dir / 'error.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
