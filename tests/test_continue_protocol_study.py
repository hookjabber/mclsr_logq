import json
from pathlib import Path
import subprocess
import sys

import pytest

import test_protocol_study
from test_protocol_study import _completed_report, _prepare
from test_recover_protocol_study import install_runner, recovery_module

inputs = test_protocol_study.inputs


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_events(path, events):
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))


def make_previous(inputs, monkeypatch, *, original_success=False):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    module = recovery_module()
    original = study['runs'][0]
    write_events(
        path.with_name('queue_0.events.jsonl'),
        [dict(original, status='running'), dict(original, status='failed', returncode=-11)],
    )
    (path.parent / 'logs').mkdir(exist_ok=True)
    (path.parent / 'logs/legacy_l00_seed1.log').write_text('Segmentation fault before training finished\n')
    if original_success:
        (path.parent / 'reports').mkdir(exist_ok=True)
        original = study['runs'][1]
        _completed_report(study, original, path.parent / 'reports/v2_l00_seed1.json')
        write_events(
            path.with_name('queue_1.events.jsonl'),
            [dict(original, status='running'), dict(original, status='complete', returncode=0)],
        )
    previous = path.parent / 'previous'
    with monkeypatch.context() as patch:
        install_runner(patch, module, path)
        execute = module.subprocess.run

        def finish_then_halt(command, **kwargs):
            result = execute(command, **kwargs)
            if '-c' not in command:
                module.halt(previous, 'old shared failure budget exhausted', 1)
            return result

        patch.setattr(module.subprocess, 'run', finish_then_halt)
        with pytest.raises(RuntimeError, match='recovery halted'):
            module.run_recovery(path, 'previous', 0)
    events_path = previous / 'queue_0.events.jsonl'
    events = read_events(events_path)
    successful = next(event for event in events if event['status'] == 'complete')
    write_events(events_path, [event for event in events if event['status'] != 'complete'])
    write_events(previous / 'queue_1.events.jsonl', [{'worker': 1, 'status': 'halted', 'error': 'old shared halt'}])
    return path, study, module, previous, successful


@pytest.fixture
def prior(inputs, monkeypatch):
    return make_previous(inputs, monkeypatch)


def trial(command, study):
    config = json.loads(Path(command[command.index('--params') + 1]).read_text())
    arm = next(name for name in study['configs'] if config['experiment_name'].startswith(f'study_{name}_'))
    return arm, int(command[command.index('--seed') + 1])


def evidence_snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def attempt_config(previous, successful):
    return previous / 'configs' / Path(successful['report']).name


def remove_report(successful):
    report = Path(successful['report'])
    for target in (report, report.with_suffix('.provenance'), report.with_suffix('.pth')):
        target.unlink()


def add_failed_attempt(study, module, previous, run, *, attempt=1):
    config = json.loads(Path(study['configs'][run['arm']]['path']).read_text())
    config['experiment_name'] += f'_previous_attempt{attempt}'
    config['path_to_recovery_manifest'] = str(previous / 'recovery.json')
    name = f'{run["arm"]}_seed{run["seed"]}_attempt{attempt}'
    config_path = previous / 'configs' / f'{name}.json'
    module.write_json(config_path, config)
    (previous / 'logs' / f'{name}.log').write_text('cudaErrorLaunchFailure\n')
    event = dict(
        run,
        attempt=attempt,
        config_sha256=module.sha256(config_path),
        report=str(previous / 'reports' / f'{name}.json'),
        status='running',
    )
    events_path = previous / f'queue_{run["index"] % 2}.events.jsonl'
    events = read_events(events_path)
    write_events(events_path, [*events[:-1], event, dict(event, status='failed', returncode=-11), events[-1]])
    return config_path


def test_continuation_workers_progress_sequentially_without_repeating_finished_test(prior, monkeypatch):
    path, study, module, previous, successful = prior
    before = evidence_snapshot(previous)
    original_before = path.with_name('queue_0.events.jsonl').read_bytes()
    calls, _ = install_runner(monkeypatch, module, path)
    worker_trials = []
    for worker in range(2):
        start = len(calls)
        result = module.run_recovery(path, 'continuation', worker, previous_recovery=previous)
        launched = [trial(command, study) for command in calls[start:]]
        expected = {
            (run['arm'], run['seed'])
            for run in study['runs']
            if run['index'] % 2 == worker and run['index'] != successful['index']
        }
        assert result['complete'] is True
        assert len(launched) == len(expected) > 1
        assert set(launched) == expected
        events = read_events(path.parent / f'continuation/queue_{worker}.events.jsonl')
        assert [event['status'] for event in events] == ['running', 'complete'] * len(launched)
        worker_trials.append(set(launched))
    assert worker_trials[0].isdisjoint(worker_trials[1])
    assert len(calls) == 14
    assert evidence_snapshot(previous) == before
    assert path.with_name('queue_0.events.jsonl').read_bytes() == original_before


def test_completed_previous_queues_with_summaries_do_not_repeat_any_test(inputs, monkeypatch):
    path = _prepare(inputs)
    module = recovery_module()
    calls, _ = install_runner(monkeypatch, module, path)
    for worker in range(2):
        module.run_recovery(path, 'previous', worker)
    assert len(calls) == 15
    before = evidence_snapshot(path.parent / 'previous')
    for worker in range(2):
        result = module.run_recovery(path, 'continuation', worker, previous_recovery=path.parent / 'previous')
        assert result['complete'] is True
    assert len(calls) == 15
    assert evidence_snapshot(path.parent / 'previous') == before


@pytest.mark.parametrize('changed', ['budget', 'provenance', 'checkpoint'])
def test_continuation_rejects_invalid_prior_report_even_without_complete_event(prior, monkeypatch, changed):
    path, _, module, previous, successful = prior
    report = Path(successful['report'])
    if changed == 'budget':
        content = json.loads(report.read_text())
        content['optimizer_updates'] = 0
        report.write_text(json.dumps(content))
    else:
        target = report.with_suffix('.provenance' if changed == 'provenance' else '.pth')
        target.write_bytes(target.read_bytes() + b'changed')
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []


@pytest.mark.parametrize('changed', ['optimizer', 'extra_key', 'experiment_name', 'path_to_recovery_manifest'])
def test_prior_config_cannot_smuggle_scientific_changes_with_matching_report(prior, monkeypatch, changed):
    path, study, module, previous, successful = prior
    config_path = attempt_config(previous, successful)
    config = json.loads(config_path.read_text())
    if changed == 'optimizer':
        config['optimizer']['lr'] = 123.0
    elif changed == 'extra_key':
        config['unexpected_scientific_override'] = True
    else:
        config[changed] = 'unrelated'
    config_path.write_text(json.dumps(config))
    digest = module.sha256(config_path)
    events_path = previous / 'queue_0.events.jsonl'
    events = read_events(events_path)
    for event in events:
        if event.get('index') == successful['index']:
            event['config_sha256'] = digest
    write_events(events_path, events)
    remove_report(successful)
    expected = dict(study, configs={successful['arm']: {'path': str(config_path), 'sha256': digest}})
    _completed_report(expected, successful, Path(successful['report']))
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []


@pytest.mark.parametrize('state', ['training_finished', 'unresolved_running', 'active_worker', 'failed_missing_log'])
def test_ambiguous_or_active_previous_attempt_blocks_all_launches(prior, monkeypatch, state):
    path, _, module, previous, successful = prior
    events_path = previous / 'queue_0.events.jsonl'
    events = read_events(events_path)
    log = previous / 'logs' / Path(successful['report']).with_suffix('.log').name
    if state == 'active_worker':
        write_events(events_path, events[:-1])
    else:
        remove_report(successful)
        if state != 'unresolved_running':
            write_events(events_path, [events[0], dict(events[0], status='failed', returncode=-11), events[-1]])
        if state == 'training_finished':
            log.write_text('Training procedure has been finished!\n' + 'output\n' * 12000 + 'Segmentation fault\n')
        elif state == 'failed_missing_log':
            log.unlink()
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []


def test_original_and_previous_attempts_share_two_attempt_limit(prior, monkeypatch):
    path, study, module, previous, _ = prior
    add_failed_attempt(study, module, previous, study['runs'][0])
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises((ValueError, RuntimeError), match='attempt|budget'):
        module.run_recovery(path, 'continuation', 1, previous_recovery=previous)
    assert calls == health == []


def test_untracked_previous_report_blocks_continuation(prior, monkeypatch):
    path, _, module, previous, successful = prior
    report = Path(successful['report'])
    (previous / 'reports/untracked.json').write_bytes(report.read_bytes())
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []


@pytest.mark.parametrize(
    'changed',
    [
        'events',
        'log',
        'config',
        'report',
        'provenance',
        'checkpoint',
        'manifest',
        'halt',
        'original_events',
        'original_log',
    ],
)
def test_prior_evidence_mutation_prevents_next_launch(prior, monkeypatch, changed):
    path, _, module, previous, successful = prior
    report = Path(successful['report'])
    target = {
        'events': previous / 'queue_0.events.jsonl',
        'log': previous / 'logs' / report.with_suffix('.log').name,
        'config': attempt_config(previous, successful),
        'report': report,
        'provenance': report.with_suffix('.provenance'),
        'checkpoint': report.with_suffix('.pth'),
        'manifest': previous / 'recovery.json',
        'halt': previous / 'HALT.json',
        'original_events': path.with_name('queue_0.events.jsonl'),
        'original_log': path.parent / 'logs/legacy_l00_seed1.log',
    }[changed]
    calls, _ = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def mutate_after_child(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command:
            target.write_bytes(target.read_bytes() + b'\n')
        return result

    monkeypatch.setattr(module.subprocess, 'run', mutate_after_child)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert len(calls) == 1


@pytest.mark.parametrize('changed', ['report', 'provenance', 'checkpoint'])
def test_original_success_evidence_is_guarded_during_continuation(inputs, monkeypatch, changed):
    path, _, module, previous, _ = make_previous(inputs, monkeypatch, original_success=True)
    report = path.parent / 'reports/v2_l00_seed1.json'
    target = report if changed == 'report' else report.with_suffix('.provenance' if changed == 'provenance' else '.pth')
    calls, _ = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def mutate_after_child(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command:
            target.write_bytes(target.read_bytes() + b'\n')
        return result

    monkeypatch.setattr(module.subprocess, 'run', mutate_after_child)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert len(calls) == 1


def test_launcher_is_prepended_and_bound_in_continuation_manifest(prior, monkeypatch, tmp_path):
    path, study, module, previous, _ = prior
    launcher = tmp_path / 'native-launcher.sh'
    launcher.write_text('#!/bin/sh\nexec "$@"\n')
    launcher.chmod(0o700)
    calls, _ = install_runner(monkeypatch, module, path)
    result = module.run_recovery(path, 'continuation', 0, previous_recovery=previous, launcher=launcher)
    assert result['complete'] is True
    assert len(calls) == 7
    for command in calls:
        assert command[:3] == [
            str(launcher),
            sys.executable,
            str(Path(study['repo_root']) / 'scripts/train_confirmatory.py'),
        ]
    manifest = (path.parent / 'continuation/recovery.json').read_text()
    assert str(launcher) in manifest
    assert module.sha256(launcher) in manifest


@pytest.mark.parametrize('when', ['startup_probe', 'first_child'])
def test_launcher_mutation_blocks_next_child(prior, monkeypatch, tmp_path, when):
    path, _, module, previous, _ = prior
    launcher = tmp_path / 'native-launcher.sh'
    launcher.write_text('#!/bin/sh\nexec "$@"\n')
    launcher.chmod(0o700)
    calls, _ = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def mutate_launcher(command, **kwargs):
        result = execute(command, **kwargs)
        if ('-c' in command) == (when == 'startup_probe'):
            launcher.write_text(launcher.read_text() + '\n')
        return result

    monkeypatch.setattr(module.subprocess, 'run', mutate_launcher)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous, launcher=launcher)
    assert len(calls) == (0 if when == 'startup_probe' else 1)


@pytest.mark.parametrize('returncode', [1, -11, 139])
def test_first_continuation_failure_halts_both_workers_without_retry(prior, monkeypatch, returncode):
    path, _, module, previous, _ = prior
    old_halt = (previous / 'HALT.json').read_bytes()
    calls, health = install_runner(monkeypatch, module, path, fail_first=True)
    execute = module.subprocess.run

    def fail_child(command, **kwargs):
        result = execute(command, **kwargs)
        return result if '-c' in command else subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(module.subprocess, 'run', fail_child)
    with pytest.raises(RuntimeError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    root = path.parent / 'continuation'
    assert (root / 'HALT.json').is_file()
    events = read_events(root / 'queue_0.events.jsonl')
    assert [event['status'] for event in events] == ['running', 'failed', 'halted']
    with pytest.raises(RuntimeError, match='halt'):
        module.run_recovery(path, 'continuation', 1, previous_recovery=previous)
    assert len(calls) == len(health) == 1
    assert (previous / 'HALT.json').read_bytes() == old_halt


@pytest.mark.parametrize('returncode', [0, -11])
def test_shared_halt_still_records_valid_child_report_without_next_launch(prior, monkeypatch, returncode):
    path, _, module, previous, _ = prior
    root = path.parent / 'continuation'
    calls, health = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run

    def halt_after_report(command, **kwargs):
        result = execute(command, **kwargs)
        if '-c' not in command:
            module.halt(root, 'other worker failed', 1)
            return subprocess.CompletedProcess(command, returncode)
        return result

    monkeypatch.setattr(module.subprocess, 'run', halt_after_report)
    with pytest.raises(RuntimeError, match='halt'):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    events = read_events(root / 'queue_0.events.jsonl')
    assert [event['status'] for event in events] == ['running', 'complete', 'halted']
    assert events[1]['returncode'] == returncode
    assert len(calls) == len(health) == 1


def test_nonzero_child_publishes_halt_before_integrity_checks_but_keeps_valid_report(prior, monkeypatch):
    path, _, module, previous, _ = prior
    root = path.parent / 'continuation'
    calls, health = install_runner(monkeypatch, module, path)
    execute = module.subprocess.run
    original_sha256 = module.sha256
    child_returned = False
    checked_halt = False

    def fail_after_report(command, **kwargs):
        nonlocal child_returned
        result = execute(command, **kwargs)
        if '-c' not in command:
            child_returned = True
            return subprocess.CompletedProcess(command, 139)
        return result

    def check_halt_before_hash(target):
        nonlocal checked_halt
        if child_returned and not checked_halt:
            assert (root / 'HALT.json').exists(), 'publish shared HALT before post-child integrity checks'
            checked_halt = True
        return original_sha256(target)

    monkeypatch.setattr(module.subprocess, 'run', fail_after_report)
    monkeypatch.setattr(module, 'sha256', check_halt_before_hash)
    with pytest.raises(RuntimeError, match='halt'):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert checked_halt
    events = read_events(root / 'queue_0.events.jsonl')
    assert [event['status'] for event in events] == ['running', 'complete', 'halted']
    assert events[1]['returncode'] == 139
    assert len(calls) == len(health) == 1


def test_missing_original_failed_log_blocks_continuation_before_launch(prior, monkeypatch):
    path, _, module, previous, _ = prior
    (path.parent / 'logs/legacy_l00_seed1.log').unlink()
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []


@pytest.mark.parametrize('changed', ['last_running', 'summary_worker', 'summary_indices'])
def test_prior_summary_cannot_authorize_running_or_mismatched_worker(prior, monkeypatch, changed):
    path, _, module, previous, successful = prior
    events_path = previous / 'queue_0.events.jsonl'
    running = read_events(events_path)[0]
    events = [running] if changed == 'last_running' else [running, successful]
    write_events(events_path, events)
    summary = {
        'worker': 1 if changed == 'summary_worker' else 0,
        'complete': True,
        'completed_indices': [0] if changed == 'summary_indices' else [successful['index']],
    }
    (previous / 'queue_0.summary.json').write_text(json.dumps(summary))
    calls, health = install_runner(monkeypatch, module, path)
    with pytest.raises(ValueError):
        module.run_recovery(path, 'continuation', 0, previous_recovery=previous)
    assert calls == health == []
