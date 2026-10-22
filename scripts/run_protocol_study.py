"""Execute one disjoint queue of a frozen study; refuse changed inputs or implicit resume."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_protocol_study import sha256, source_hashes, write_json


def validate_study(path):
    path = Path(path).resolve()
    if (path.parent / '.preparation-incomplete').exists():
        raise ValueError('study preparation is incomplete')
    study = json.loads(path.read_text())
    if study['source_sha256'] != source_hashes(study['repo_root']):
        raise ValueError('source changed since study preparation')
    for record in study['configs'].values():
        if sha256(record['path']) != record['sha256']:
            raise ValueError(f'configuration hash mismatch: {record["path"]}')
    for record in study['data'].values():
        for name in (
            'preprocessing.json',
            'preprocessing_manifest.json',
            'manifest.json',
            '.preprocessing-in-progress',
        ):
            if (Path(record['root']) / name).exists() != (name in record['files_sha256']):
                raise ValueError(f'dataset metadata presence changed: {record["root"]}/{name}')
        for name, expected in record['files_sha256'].items():
            if sha256(Path(record['root']) / name) != expected:
                raise ValueError(f'dataset hash mismatch: {record["root"]}/{name}')
    return study


def partition_runs(study, worker, workers):
    if workers != study['workers']:
        raise ValueError('workers must match the frozen study partition')
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('worker must be in [0, workers)')
    return [run for run in study['runs'] if run['index'] % workers == worker]


def validate_report(study, run, path):
    report = json.loads(Path(path).read_text())
    config_record = study['configs'][run['arm']]
    config = json.loads(Path(config_record['path']).read_text())
    expected = {
        'experiment': f'{config["experiment_name"]}_confirm_seed{run["seed"]}',
        'seed': run['seed'],
        'step_limit': study['steps'],
        'optimizer_updates': study['steps'],
        'params_sha256': config_record['sha256'],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('report identity/config/update budget mismatch')
    provenance = report['provenance']
    if sha256(provenance['path']) != provenance['sha256']:
        raise ValueError('report provenance hash mismatch')
    manifest = json.loads(Path(provenance['path']).read_text())
    if (
        manifest['seed'] != run['seed']
        or manifest['selection'] != study['selection']
        or manifest['step_limit'] != study['steps']
        or manifest['params']['sha256'] != config_record['sha256']
    ):
        raise ValueError('report provenance identity mismatch')
    for name, expected_hash in study['source_sha256'].items():
        if manifest['source']['files'].get(name, {}).get('sha256') != expected_hash:
            raise ValueError(f'report source hash mismatch: {name}')
    if set(report['results']) != set(study['selection']):
        raise ValueError('report checkpoint selection mismatch')
    for metric, result in report['results'].items():
        if result['selection_metric'] != metric or not 0 <= result['step'] < study['steps']:
            raise ValueError('report checkpoint metadata mismatch')
        if not math.isfinite(result['validation_value']) or set(result['test']) != {'ndcg@20', 'recall@1000'}:
            raise ValueError('report metrics mismatch')
        if not all(math.isfinite(value) for value in result['test'].values()):
            raise ValueError('report contains nonfinite test metrics')
        checkpoint = Path(study['repo_root']) / result['checkpoint']
        if sha256(checkpoint) != result['checkpoint_sha256']:
            raise ValueError('report checkpoint hash mismatch')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    study = validate_study(args.study)
    runs = partition_runs(study, args.worker, args.workers)
    root = Path(study['root'])
    logs = root / 'logs'
    reports = root / 'reports'
    logs.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    digest = sha256(args.study)
    write_json(
        root / f'queue_{args.worker}.reservation.json',
        {
            'worker': args.worker,
            'workers': args.workers,
            'pid': os.getpid(),
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'study_sha256': digest,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'runs': runs,
        },
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(study['repo_root']) / 'src'))
    events = root / f'queue_{args.worker}.events.jsonl'
    with events.open('x', buffering=1) as status:
        for run in runs:
            if sha256(args.study) != digest:
                raise ValueError('study manifest changed while queue was running')
            validate_study(args.study)
            name = f'{run["arm"]}_seed{run["seed"]}'
            event = dict(run, status='running', time_utc=datetime.now(timezone.utc).isoformat())
            status.write(json.dumps(event) + '\n')
            print(f'START {name}', flush=True)
            command = [
                sys.executable,
                str(Path(study['repo_root']) / 'scripts/train_confirmatory.py'),
                '--params',
                study['configs'][run['arm']]['path'],
                '--seed',
                str(run['seed']),
                '--max-steps',
                str(study['steps']),
                '--select',
                *study['selection'],
                '--output',
                str(reports / f'{name}.json'),
            ]
            with (logs / f'{name}.log').open('x') as log:
                result = subprocess.run(
                    command, cwd=study['repo_root'], env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            returncode = result.returncode
            if returncode == 0:
                try:
                    if sha256(args.study) != digest:
                        raise ValueError('study manifest changed during training')
                    validate_study(args.study)
                    validate_report(study, run, reports / f'{name}.json')
                except (OSError, ValueError, KeyError, TypeError) as error:
                    event['error'] = str(error)
                    returncode = 1
            event.update(
                status='complete' if returncode == 0 else 'failed',
                returncode=returncode,
                time_utc=datetime.now(timezone.utc).isoformat(),
            )
            status.write(json.dumps(event) + '\n')
            print(f'{event["status"].upper()} {name} exit={returncode}', flush=True)
            if returncode:
                raise SystemExit(returncode)
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
