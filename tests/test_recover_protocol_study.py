import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import pytest

import test_protocol_study
from test_protocol_study import _completed_report, _prepare

inputs = test_protocol_study.inputs


def recovery_module():
    assert importlib.util.find_spec('recover_protocol_study') is not None
    return __import__('recover_protocol_study')


def install_runner(monkeypatch, module, path, *, fail_first=False, unhealthy=False, mutate=False):
    study = json.loads(path.read_text())
    calls = []
    health = []

    def execute(command, **kwargs):
        if '-c' in command:
            health.append(command)
            return subprocess.CompletedProcess(command, 1 if unhealthy and len(health) > 1 else 0)
        calls.append(command)
        if fail_first and len(calls) == 1:
            kwargs['stdout'].write('cudaErrorLaunchFailure: unspecified launch failure\n')
            kwargs['stdout'].flush()
            return subprocess.CompletedProcess(command, 1)
        config_path = Path(command[command.index('--params') + 1])
        config = json.loads(config_path.read_text())
        arm = next(name for name in study['configs'] if config['experiment_name'].startswith(f'study_{name}_'))
        run = {'arm': arm, 'seed': int(command[command.index('--seed') + 1])}
        expected = dict(study, configs={arm: {'path': str(config_path), 'sha256': module.sha256(config_path)}})
        _completed_report(expected, run, Path(command[command.index('--output') + 1]))
        if mutate:
            path.write_text(path.read_text() + '\n')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', execute)
    return calls, health


def test_recovery_runs_all_queued_trials_in_sequence_without_overwriting(inputs, monkeypatch):
    path = _prepare(inputs)
    original = path.read_bytes()
    module = recovery_module()
    calls, health = install_runner(monkeypatch, module, path)
    result = module.run_recovery(path, 'recovery_01', 0)
    assert result['complete'] is True
    assert len(calls) == 8 and len(health) == 1
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        module.run_recovery(path, 'recovery_01', 0)


@pytest.mark.parametrize('returncode', [0, -11])
@pytest.mark.parametrize('completed_before_halt', [1, 8])
def test_valid_report_is_completed_after_shared_halt(inputs, monkeypatch, capsys, returncode, completed_before_halt):
    path = _prepare(inputs)
    module = recovery_module()
    root = path.parent / 'recovery_01'
    calls, health = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def halt_after_report(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command and len(calls) == completed_before_halt:
            module.halt(root, 'other worker stopped new launches', 1)
            return subprocess.CompletedProcess(command, returncode)
        return result

    monkeypatch.setattr(module.subprocess, 'run', halt_after_report)
    if completed_before_halt == 8:
        result = module.run_recovery(path, 'recovery_01', 0)
        assert result['complete'] is True
        assert result['completed_indices'] == list(range(0, 15, 2))
        assert json.loads((root / 'queue_0.summary.json').read_text()) == result
    else:
        with pytest.raises(RuntimeError, match='recovery halted'):
            module.run_recovery(path, 'recovery_01', 0)
        assert not (root / 'queue_0.summary.json').exists()
    events = [json.loads(line) for line in (root / 'queue_0.events.jsonl').read_text().splitlines()]
    completed = [event for event in events if event['status'] == 'complete']
    assert len(completed) == completed_before_halt
    assert completed[-1]['index'] == (completed_before_halt - 1) * 2
    assert completed[-1]['returncode'] == returncode
    assert len(calls) == completed_before_halt and len(health) == 1
    assert capsys.readouterr().out.count('COMPLETE ') == completed_before_halt
    assert json.loads((root / 'HALT.json').read_text())['worker'] == 1


@pytest.mark.parametrize(
    ('changed', 'message'),
    [
        ('source', 'source changed'),
        ('study', 'study manifest changed'),
        ('supervisor', 'recovery supervisor/manifest changed'),
        ('recovery', 'recovery supervisor/manifest changed'),
        ('original_evidence', 'original recovery evidence changed'),
        ('original_config', 'configuration hash mismatch'),
        ('attempt_config', 'attempt configuration changed'),
        ('dataset', 'dataset hash mismatch'),
        ('provenance', 'report provenance hash mismatch'),
        ('checkpoint', 'report checkpoint hash mismatch'),
        ('budget', 'report identity/config/update budget mismatch'),
    ],
)
def test_changed_inputs_with_shared_halt_cannot_record_complete(inputs, monkeypatch, changed, message):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    original_events = path.with_name('queue_1.events.jsonl')
    original_events.write_text(json.dumps(dict(study['runs'][1], status='failed', returncode=-11)) + '\n')
    module = recovery_module()
    root = path.parent / 'recovery_01'
    calls, health = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def change_inputs_after_report(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command:
            report_path = Path(command[command.index('--output') + 1])
            if changed == 'source':
                changed_hashes = dict(study['source_sha256'])
                changed_hashes['scripts/train_confirmatory.py'] = '0' * 64
                monkeypatch.setattr(__import__('run_protocol_study'), 'source_hashes', lambda repo: changed_hashes)
            elif changed == 'supervisor':
                original_sha256 = module.sha256
                monkeypatch.setattr(
                    module, 'sha256', lambda target: '0' * 64 if target == module.__file__ else original_sha256(target)
                )
            elif changed == 'budget':
                report = json.loads(report_path.read_text())
                report['optimizer_updates'] = 0
                report_path.write_text(json.dumps(report))
            else:
                target = {
                    'study': path,
                    'recovery': root / 'recovery.json',
                    'original_evidence': original_events,
                    'original_config': Path(study['configs']['legacy_l00']['path']),
                    'attempt_config': Path(command[command.index('--params') + 1]),
                    'dataset': Path(study['data']['legacy']['root']) / 'train_sasrec.txt',
                    'provenance': report_path.with_suffix('.provenance'),
                    'checkpoint': report_path.with_suffix('.pth'),
                }[changed]
                target.write_bytes(target.read_bytes() + b'\n')
            module.halt(root, 'other worker stopped new launches', 1)
        return result

    monkeypatch.setattr(module.subprocess, 'run', change_inputs_after_report)
    with pytest.raises(ValueError, match=message):
        module.run_recovery(path, 'recovery_01', 0)
    events = [json.loads(line) for line in (root / 'queue_0.events.jsonl').read_text().splitlines()]
    assert [event['status'] for event in events] == ['running', 'halted']
    assert len(calls) == len(health) == 1
    assert not (root / 'queue_0.summary.json').exists()
    assert json.loads((root / 'HALT.json').read_text())['worker'] == 1


@pytest.mark.parametrize('before', ['startup', 'child', 'failure_probe'])
def test_shared_halt_blocks_subprocess_launches(inputs, monkeypatch, before):
    path = _prepare(inputs)
    module = recovery_module()
    root = path.parent / 'recovery_01'
    calls, health = install_runner(monkeypatch, module, path, fail_first=before == 'failure_probe')
    write_json = module.write_json

    def halt_after_write(target, content):
        write_json(target, content)
        if (before == 'child' and target.parent == root / 'configs') or (
            before == 'failure_probe' and target.parent == root / 'failures'
        ):
            module.halt(root, 'other worker stopped new launches', 1)

    monkeypatch.setattr(module, 'write_json', halt_after_write)
    if before == 'startup':
        root.mkdir()
        module.halt(root, 'other worker stopped new launches', 1)
    with pytest.raises(RuntimeError, match='recovery halted'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == (1 if before == 'failure_probe' else 0)
    assert len(health) == (0 if before == 'startup' else 1)
    events = [json.loads(line) for line in (root / 'queue_0.events.jsonl').read_text().splitlines()]
    assert not any(event['status'] == 'complete' for event in events)
    assert json.loads((root / 'HALT.json').read_text())['worker'] == 1


@pytest.mark.parametrize('probe_number', [1, 2])
def test_shared_halt_during_probe_blocks_next_child(inputs, monkeypatch, probe_number):
    path = _prepare(inputs)
    module = recovery_module()
    root = path.parent / 'recovery_01'
    calls, health = install_runner(monkeypatch, module, path, fail_first=True)
    execute = module.subprocess.run

    def halt_during_probe(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' in command and len(health) == probe_number:
            module.halt(root, 'other worker stopped new launches', 1)
        return result

    monkeypatch.setattr(module.subprocess, 'run', halt_during_probe)
    with pytest.raises(RuntimeError, match='recovery halted'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == probe_number - 1
    assert len(health) == probe_number


def test_shared_halt_blocks_pending_retry_after_recording_success(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    root = path.parent / 'recovery_01'
    calls, health = install_runner(monkeypatch, module, path, fail_first=True)
    execute = module.subprocess.run

    def halt_before_retry(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command and len(calls) == 8:
            module.halt(root, 'other worker stopped new launches', 1)
        return result

    monkeypatch.setattr(module.subprocess, 'run', halt_before_retry)
    with pytest.raises(RuntimeError, match='recovery halted'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == 8 and len(health) == 2
    events = [json.loads(line) for line in (root / 'queue_0.events.jsonl').read_text().splitlines()]
    assert sum(event['status'] == 'complete' for event in events) == 7
    assert not any(event.get('attempt') == 2 for event in events)


def test_old_failed_runs_are_repeated_after_unstarted_runs(inputs, monkeypatch):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    failed = dict(study['runs'][0], status='failed', returncode=-11)
    old_events = path.with_name('queue_0.events.jsonl')
    old_events.write_text(json.dumps(failed) + '\n')
    module = recovery_module()
    calls, _ = install_runner(monkeypatch, module, path)
    module.run_recovery(path, 'recovery_01', 0)
    names = [Path(command[command.index('--params') + 1]).name for command in calls]
    assert names[0].startswith('legacy_target_l10_seed1')
    assert names[-1].startswith('legacy_l00_seed1')
    assert json.loads(old_events.read_text()) == failed


def test_transient_failure_does_not_drop_remaining_runs_and_gets_one_retry(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls, health = install_runner(monkeypatch, module, path, fail_first=True)
    result = module.run_recovery(path, 'recovery_01', 0)
    assert result['complete'] is True
    assert len(calls) == 9 and len(health) == 2
    assert calls[0][calls[0].index('--params') + 1] != calls[-1][calls[-1].index('--params') + 1]
    events = [json.loads(line) for line in (path.parent / 'recovery_01/queue_0.events.jsonl').read_text().splitlines()]
    assert any(event['status'] == 'failed' for event in events)
    assert sum(event['status'] == 'complete' for event in events) == 8


def test_unhealthy_gpu_stops_recovery_instead_of_launching_more_jobs(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls, _ = install_runner(monkeypatch, module, path, fail_first=True, unhealthy=True)
    with pytest.raises(RuntimeError, match='GPU health'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == 1


def test_manifest_change_during_run_stops_recovery(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls, _ = install_runner(monkeypatch, module, path, mutate=True)
    with pytest.raises(ValueError, match='study manifest changed'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == 1


def test_recovery_refuses_original_running_queue(inputs):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    path.with_name('queue_0.events.jsonl').write_text(json.dumps(dict(study['runs'][0], status='running')) + '\n')
    with pytest.raises(ValueError, match='running'):
        recovery_module().run_recovery(path, 'recovery_01', 0)


def test_code_failure_is_not_retried_or_ignored(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls = []

    def execute(command, **kwargs):
        if '-c' in command:
            return subprocess.CompletedProcess(command, 0)
        calls.append(command)
        kwargs['stdout'].write('ValueError: invalid model configuration\n')
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='unclassified'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == 1


def test_failure_after_training_is_not_repeated_with_another_test_evaluation(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()

    def execute(command, **kwargs):
        if '-c' in command:
            return subprocess.CompletedProcess(command, 0)
        kwargs['stdout'].write(
            'Training procedure has been finished!\n' + 'diagnostic output\n' * 5000 + 'cudaErrorLaunchFailure\n'
        )
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='test evaluation'):
        module.run_recovery(path, 'recovery_01', 0)


def test_repeated_infrastructure_failures_open_shared_circuit_breaker(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls = []

    def execute(command, **kwargs):
        if '-c' in command:
            return subprocess.CompletedProcess(command, 0)
        calls.append(command)
        kwargs['stdout'].write('cudaErrorLaunchFailure\n')
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='infrastructure failure budget'):
        module.run_recovery(path, 'recovery_01', 0)
    assert len(calls) == 2
    assert (path.parent / 'recovery_01/HALT.json').exists()


def test_shared_manifest_is_published_atomically_and_rejects_conflicts(tmp_path):
    module = recovery_module()
    assert hasattr(module, 'ensure_manifest')
    path = tmp_path / 'recovery.json'
    content = {'schedule': list(range(500))}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(module.ensure_manifest, path, content) for _ in range(2)]
        for future in futures:
            future.result()
    assert json.loads(path.read_text()) == content
    with pytest.raises(ValueError, match='conflicts'):
        module.ensure_manifest(path, {'different': True})
