"""Recover a frozen study with isolated attempts and bounded infrastructure retries.

For an existing frozen checkout, run a copy outside its source tree with that
checkout's src and scripts on PYTHONPATH. Model code and original artifacts stay
immutable. Failed training restarts from its original seed, never from best weights.
Shared HALT prevents further launches; already running children may finish.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from run_protocol_study import partition_runs, sha256, validate_report, validate_study, write_json

GPU_PROBE = """
import torch
assert torch.cuda.device_count() == 1, 'expected one allocated GPU'
torch.set_num_threads(1)
values = torch.randn(512, 512, device='cuda', requires_grad=True)
matrix = torch.eye(512, device='cuda').to_sparse()
for step in range(32):
    values.grad = None
    loss = (torch.sparse.mm(matrix, values) @ values.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
print('GPU_HEALTH_PASS', torch.cuda.get_device_name(0), flush=True)
"""


def log_tail(path):
    if not Path(path).exists():
        return ''
    with Path(path).open('rb') as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - 65536))
        return source.read().decode('utf-8', errors='replace')


def training_finished(path):
    if not Path(path).exists():
        return False
    with Path(path).open(errors='replace') as source:
        return any('Training procedure has been finished!' in line for line in source)


def original_state(study):
    root = Path(study['root'])
    states = {}
    hashes = {}
    for worker in range(study['workers']):
        path = root / f'queue_{worker}.events.jsonl'
        if path.exists():
            hashes[str(path)] = sha256(path)
            for line in path.read_text().splitlines():
                event = json.loads(line)
                states[event['index']] = event['status']
    if 'running' in states.values():
        raise ValueError('an original queue is still recorded as running; reconcile it before recovery')
    for run in study['runs']:
        name = f'{run["arm"]}_seed{run["seed"]}'
        report = root / 'reports' / f'{name}.json'
        if report.exists():
            validate_report(study, run, report)
            hashes[str(report)] = sha256(report)
            states[run['index']] = 'complete'
        elif states.get(run['index']) == 'complete':
            raise ValueError('original completion has no valid report')
        elif training_finished(root / 'logs' / f'{name}.log'):
            raise RuntimeError('original run may have reached test evaluation; reconcile without retraining')
    return states, hashes


def halt(root, reason, worker):
    try:
        write_json(
            root / 'HALT.json', {'reason': reason, 'worker': worker, 'time_utc': datetime.now(timezone.utc).isoformat()}
        )
    except FileExistsError:
        pass


def ensure_manifest(path, content):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    write_json(temporary, content)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != json.loads(json.dumps(content)):
                raise ValueError('recovery manifest conflicts with existing recovery')
    finally:
        temporary.unlink()


def report_evidence(study, report, hashes):
    provenance = report['provenance']
    hashes[str(Path(provenance['path']).resolve())] = provenance['sha256']
    for result in report['results'].values():
        checkpoint = (Path(study['repo_root']) / result['checkpoint']).resolve()
        hashes[str(checkpoint)] = result['checkpoint_sha256']
    for path, expected in report.get('artifact_sha256', {}).items():
        if sha256(path) != expected:
            raise ValueError('previous report artifact hash mismatch')
        hashes[str(Path(path).resolve())] = expected


def evidence_files(root):
    return sorted(str(path.resolve()) for path in Path(root).rglob('*') if path.is_file())


def reconcile_previous(study_path, study, previous_recovery):
    root = Path(previous_recovery).resolve()
    manifest_path = root / 'recovery.json'
    manifest = json.loads(manifest_path.read_text())
    if (
        Path(manifest['original_study']).resolve() != Path(study_path).resolve()
        or manifest['original_study_sha256'] != sha256(study_path)
        or manifest['workers'] != study['workers']
        or manifest.get('previous_recovery')
    ):
        raise ValueError('previous recovery manifest does not describe this first-generation recovery')
    states, hashes = original_state(study)
    attempts = {index: 1 for index in states}
    for path, expected in manifest['original_event_report_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('previous recovery original evidence changed')
        hashes[path] = expected
    previous_files = evidence_files(root)
    hashes.update({path: sha256(path) for path in previous_files})
    for run in study['runs']:
        name = f'{run["arm"]}_seed{run["seed"]}'
        log = Path(study['root']) / 'logs' / f'{name}.log'
        if states.get(run['index']) == 'failed' and not log.is_file():
            raise ValueError('original failed attempt is missing its log; test evaluation cannot be excluded')
        if log.exists():
            hashes[str(log)] = sha256(log)
        if states.get(run['index']) == 'complete':
            report = validate_report(study, run, Path(study['root']) / 'reports' / f'{name}.json')
            report_evidence(study, report, hashes)
    known_runs = {run['index']: run for run in study['runs']}
    records = {}
    for worker in range(study['workers']):
        path = root / f'queue_{worker}.events.jsonl'
        if not path.exists():
            if (root / f'queue_{worker}.reservation.json').exists():
                raise ValueError('previous recovery worker has unresolved reservation')
            continue
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if not events:
            raise ValueError('previous recovery worker is unresolved/running')
        if events[-1]['status'] != 'halted':
            summary_path = root / f'queue_{worker}.summary.json'
            if events[-1]['status'] != 'complete' or not summary_path.is_file():
                raise ValueError('previous recovery worker is still running or unresolved')
            summary = json.loads(summary_path.read_text())
            expected_indices = {
                run['index']
                for run in study['runs']
                if run['index'] % study['workers'] == worker and states.get(run['index']) != 'complete'
            }
            completed_indices = summary.get('completed_indices', [])
            final_events = {event['index']: event['status'] for event in events if 'index' in event}
            if (
                summary.get('complete') is not True
                or summary.get('worker') != worker
                or set(completed_indices) != expected_indices
                or len(completed_indices) != len(expected_indices)
                or final_events != dict.fromkeys(expected_indices, 'complete')
            ):
                raise ValueError('previous recovery completion summary is inconsistent')
        for event in events:
            if event['status'] == 'halted':
                continue
            run = known_runs.get(event['index'])
            if (
                run is None
                or any(event.get(key) != value for key, value in run.items())
                or run['index'] % study['workers'] != worker
                or not isinstance(event.get('attempt'), int)
                or event['attempt'] < 1
            ):
                raise ValueError('previous recovery event identity mismatch')
            key = (run['index'], event['attempt'])
            if event['status'] == 'running':
                if key in records:
                    raise ValueError('duplicate previous recovery attempt')
                records[key] = event
                attempts[run['index']] = attempts.get(run['index'], 0) + 1
            elif event['status'] in ('complete', 'failed'):
                start = records.get(key)
                if (
                    start is None
                    or start['status'] != 'running'
                    or any(event.get(field) != start.get(field) for field in ('config_sha256', 'report'))
                ):
                    raise ValueError('previous recovery attempt event mismatch')
                records[key] = event
            else:
                raise ValueError('unknown previous recovery event status')
    expected_reports = set()
    for (index, attempt), event in records.items():
        run = known_runs[index]
        name = f'{run["arm"]}_seed{run["seed"]}_attempt{attempt}'
        config_path = root / 'configs' / f'{name}.json'
        report_path = root / 'reports' / f'{name}.json'
        expected_reports.add(report_path.resolve())
        if Path(event['report']).resolve() != report_path.resolve() or sha256(config_path) != event['config_sha256']:
            raise ValueError('previous recovery configuration/report hash or path mismatch')
        config = json.loads(config_path.read_text())
        original = json.loads(Path(study['configs'][run['arm']]['path']).read_text())
        expected_config = dict(
            original,
            experiment_name=f'{original["experiment_name"]}_{manifest["name"]}_attempt{attempt}',
            path_to_recovery_manifest=str(manifest_path),
        )
        if config != expected_config:
            raise ValueError('previous recovery configuration changed beyond namespace')
        if report_path.exists():
            expected = dict(study, configs={run['arm']: {'path': str(config_path), 'sha256': sha256(config_path)}})
            report = validate_report(expected, run, report_path)
            report_evidence(study, report, hashes)
            if states.get(index) == 'complete':
                raise ValueError('multiple completed reports for one run; reconcile without selection')
            states[index] = 'complete'
        else:
            log = root / 'logs' / f'{name}.log'
            if training_finished(log):
                raise RuntimeError('previous run may have reached test evaluation; do not repeat')
            if event['status'] != 'failed' or not log.exists():
                raise ValueError('previous recovery attempt is running/unresolved or missing a completed report/log')
            if event.get('returncode') != -11 and not any(
                marker in log_tail(log)
                for marker in (
                    'cudaErrorLaunchFailure',
                    'unspecified launch failure',
                    'CUDA error: an illegal memory access',
                )
            ):
                raise ValueError('previous failure is unclassified; manual reconciliation required')
            if states.get(index) == 'complete':
                raise ValueError('a previous attempt follows a completed result')
            states[index] = 'failed'
    if any(path.resolve() not in expected_reports for path in (root / 'reports').glob('*.json')):
        raise ValueError('previous recovery contains an untracked report')
    if any(count >= 2 and states.get(index) != 'complete' for index, count in attempts.items()):
        raise ValueError('previous run has exhausted its two total attempts')
    return states, hashes, attempts, previous_files


def run_recovery(study_path, recovery_name, worker, workers=2, *, previous_recovery=None, launcher=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', recovery_name):
        raise ValueError('recovery name must be a safe directory/experiment suffix')
    study_path = Path(study_path).resolve()
    study = validate_study(study_path)
    partition = partition_runs(study, worker, workers)
    states, old_hashes = original_state(study)
    prior_attempts = {}
    previous_files = []
    if previous_recovery is not None:
        previous_recovery = Path(previous_recovery).resolve()
        states, old_hashes, prior_attempts, previous_files = reconcile_previous(study_path, study, previous_recovery)
        if previous_recovery == Path(study['root']) / recovery_name:
            raise ValueError('continuation needs a new recovery namespace')
    launcher_record = None
    if launcher is not None:
        launcher = Path(launcher).resolve()
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise ValueError('launcher must be an executable file')
        launcher_record = {'path': str(launcher), 'sha256': sha256(launcher)}
    runs = [run for run in partition if states.get(run['index']) != 'complete']
    runs.sort(key=lambda run: (states.get(run['index']) == 'failed', run['index']))
    root = Path(study['root']) / recovery_name
    root.mkdir(exist_ok=True)
    for name in ('configs', 'logs', 'reports', 'failures'):
        (root / name).mkdir(exist_ok=True)
    manifest_path = root / 'recovery.json'
    study_digest = sha256(study_path)
    supervisor_digest = sha256(__file__)
    manifest = {
        'schema_version': 1,
        'original_study': str(study_path),
        'original_study_sha256': study_digest,
        'original_event_report_sha256': old_hashes,
        'original_status': states,
        'supervisor_sha256': supervisor_digest,
        'gpu_probe_sha256': hashlib.sha256(GPU_PROBE.encode()).hexdigest(),
        'workers': workers,
        'name': recovery_name,
        'max_new_infrastructure_failures': 2,
        'halt_policy': 'prevent further launches; already running children may finish',
        'config_changes': ['experiment_name', 'path_to_recovery_manifest'],
        'restart_policy': 'original seed from scratch; failed original runs last; at most two total attempts per run',
        'retry_classification': 'SIGSEGV or CUDA launch/illegal-address failure before training completion',
    }
    if previous_recovery is not None:
        manifest.update(
            previous_recovery=str(previous_recovery),
            previous_files=previous_files,
            prior_attempts=prior_attempts,
            remaining_runs=[run for run in study['runs'] if states.get(run['index']) != 'complete'],
            max_new_infrastructure_failures=1,
            restart_policy='at most two total scientific attempts across original and prior recovery; no automatic retry',
        )
    if launcher_record is not None:
        manifest['launcher'] = launcher_record
    ensure_manifest(manifest_path, manifest)
    manifest_digest = sha256(manifest_path)
    write_json(
        root / f'queue_{worker}.reservation.json',
        {
            'worker': worker,
            'pid': os.getpid(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'runs': runs,
            'recovery_sha256': manifest_digest,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
        },
    )
    environment = dict(
        os.environ, PYTHONPATH=str(Path(study['repo_root']) / 'src'), PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1'
    )

    def guard_integrity():
        if sha256(study_path) != study_digest:
            raise ValueError('study manifest changed during recovery')
        if sha256(manifest_path) != manifest_digest or sha256(__file__) != supervisor_digest:
            raise ValueError('recovery supervisor/manifest changed')
        for path, digest in old_hashes.items():
            if sha256(path) != digest:
                raise ValueError('original recovery evidence changed')
        if previous_recovery is not None and evidence_files(previous_recovery) != previous_files:
            raise ValueError('previous recovery evidence inventory changed')
        if launcher_record is not None and sha256(launcher) != launcher_record['sha256']:
            raise ValueError('launcher changed during recovery')
        validate_study(study_path)

    def guard_launch():
        guard_integrity()
        if (root / 'HALT.json').exists():
            raise RuntimeError('recovery halted: ' + (root / 'HALT.json').read_text())

    def health(label):
        guard_launch()
        with (root / 'logs' / f'gpu_worker{worker}_{label}.log').open('x') as output:
            result = subprocess.run(
                [sys.executable, '-c', GPU_PROBE], env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=120
            )
        if result.returncode:
            raise RuntimeError('GPU health check failed; no further training is safe')

    with (root / f'queue_{worker}.events.jsonl').open('x', buffering=1) as events:

        def record(event):
            events.write(json.dumps(dict(event, time_utc=datetime.now(timezone.utc).isoformat())) + '\n')
            events.flush()
            os.fsync(events.fileno())

        completed = []
        retries = []
        attempts = [(run, prior_attempts.get(run['index'], 0) + 1) for run in runs]
        print(f'QUEUE_PLANNED worker={worker} trials={len(runs)}', flush=True)
        try:
            health('startup')
            while attempts or retries:
                if not attempts:
                    attempts, retries = retries, []
                run, attempt = attempts.pop(0)
                guard_launch()
                name = f'{run["arm"]}_seed{run["seed"]}_attempt{attempt}'
                config = copy.deepcopy(json.loads(Path(study['configs'][run['arm']]['path']).read_text()))
                config['experiment_name'] += f'_{recovery_name}_attempt{attempt}'
                config['path_to_recovery_manifest'] = str(manifest_path)
                config_path = root / 'configs' / f'{name}.json'
                write_json(config_path, config)
                config_digest = sha256(config_path)
                expected_study = dict(study, configs={run['arm']: {'path': str(config_path), 'sha256': config_digest}})
                report = root / 'reports' / f'{name}.json'
                log = root / 'logs' / f'{name}.log'
                event = dict(run, attempt=attempt, config_sha256=config_digest, report=str(report), status='running')
                record(event)
                print(f'START {name}', flush=True)
                command = [
                    sys.executable,
                    str(Path(study['repo_root']) / 'scripts/train_confirmatory.py'),
                    '--params',
                    str(config_path),
                    '--seed',
                    str(run['seed']),
                    '--max-steps',
                    str(study['steps']),
                    '--select',
                    *study['selection'],
                    '--output',
                    str(report),
                ]
                if launcher is not None:
                    command.insert(0, str(launcher))
                with log.open('x') as output:
                    guard_launch()
                    result = subprocess.run(
                        command, cwd=study['repo_root'], env=environment, stdout=output, stderr=subprocess.STDOUT
                    )
                if previous_recovery is not None and result.returncode != 0:
                    halt(root, f'continuation child failed with exit {result.returncode}; no further launches', worker)
                if sha256(config_path) != config_digest:
                    raise ValueError('attempt configuration changed during training')
                if result.returncode == 0 or report.exists():
                    guard_integrity()
                    validate_report(expected_study, run, report)
                    completed.append(run['index'])
                    record(dict(event, status='complete', returncode=result.returncode))
                    print(f'COMPLETE {name}', flush=True)
                    continue
                record(dict(event, status='failed', returncode=result.returncode))
                print(f'FAILED {name} exit={result.returncode}', flush=True)
                guard_launch()
                tail = log_tail(log)
                if training_finished(log):
                    raise RuntimeError('run may have reached test evaluation; do not automatically repeat it')
                if previous_recovery is not None:
                    raise RuntimeError(f'continuation child failed with exit {result.returncode}; no automatic retry')
                if result.returncode != -11 and not any(
                    marker in tail
                    for marker in (
                        'cudaErrorLaunchFailure',
                        'unspecified launch failure',
                        'CUDA error: an illegal memory access',
                    )
                ):
                    raise RuntimeError('unclassified training failure; manual diagnosis required')
                for slot in range(1, 3):
                    try:
                        write_json(root / 'failures' / f'{slot}.json', dict(event, returncode=result.returncode))
                        break
                    except FileExistsError:
                        continue
                if slot >= 2:
                    raise RuntimeError('shared infrastructure failure budget exhausted; no further launches')
                health(name)
                if attempt == 1 and states.get(run['index']) != 'failed':
                    retries.append((run, 2))
                else:
                    raise RuntimeError('run retry budget exhausted')
            summary = {'worker': worker, 'complete': len(completed) == len(runs), 'completed_indices': completed}
            write_json(root / f'queue_{worker}.summary.json', summary)
            print('QUEUE_COMPLETE', flush=True)
            return summary
        except Exception as error:
            halt(root, str(error), worker)
            record({'status': 'halted', 'error': str(error), 'worker': worker})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--recovery', required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--previous-recovery', type=Path)
    parser.add_argument('--launcher', type=Path)
    args = parser.parse_args()
    run_recovery(
        args.study,
        args.recovery,
        args.worker,
        args.workers,
        previous_recovery=args.previous_recovery,
        launcher=args.launcher,
    )


if __name__ == '__main__':
    main()
