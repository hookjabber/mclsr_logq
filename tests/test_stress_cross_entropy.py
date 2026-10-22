import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stress_cross_entropy.py'


@pytest.fixture
def diagnostic():
    assert SCRIPT.is_file(), 'Standalone cross-entropy diagnostic is not implemented'
    spec = importlib.util.spec_from_file_location('stress_cross_entropy', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_args(diagnostic, output, *extra):
    return diagnostic.parse_args(
        ['--output-dir', str(output), '--device', 'cpu', '--batch-size', '3', '--iterations', '2', *extra]
    )


def read_events(output):
    return [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]


def test_defaults_require_new_output_and_explicit_cpu_opt_in(diagnostic, tmp_path):
    with pytest.raises(SystemExit):
        diagnostic.parse_args([])
    args = diagnostic.parse_args(['--output-dir', str(tmp_path / 'new')])
    assert args.device == 'cuda'
    assert args.batch_size == 128
    assert args.iterations == 100
    assert args.max_seconds == 60
    assert args.item_sizes == []


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('--iterations', '0'),
        ('--iterations', '1000001'),
        ('--iterations', '1.5'),
        ('--max-seconds', '0'),
        ('--max-seconds', '-1'),
        ('--max-seconds', '28801'),
        ('--max-seconds', 'nan'),
        ('--max-seconds', 'inf'),
        ('--seed', '-1'),
        ('--seed', str(2**63)),
        ('--batch-size', '0'),
        ('--batch-size', '129'),
        ('--item-sizes', '0'),
        ('--item-sizes', '2561'),
        ('--progress-every', '0'),
        ('--device', 'mps'),
        ('--device', 'cuda:-1'),
        ('--device', 'cpu:0'),
    ],
)
def test_parser_rejects_unbounded_or_invalid_values(diagnostic, tmp_path, option, value):
    with pytest.raises(SystemExit) as failure:
        diagnostic.parse_args(['--output-dir', str(tmp_path / 'new'), option, value])
    assert failure.value.code == 2


def test_tiny_cpu_rounds_forward_backward_and_finite_results(diagnostic, tmp_path, monkeypatch):
    output = tmp_path / 'success'
    forward = diagnostic.functional.cross_entropy
    backward_shapes = []

    def observed_forward(scores, targets):
        scores.register_hook(lambda gradient: backward_shapes.append(list(gradient.shape)))
        assert scores.requires_grad
        assert targets.dtype == torch.int64
        assert torch.equal(targets, torch.arange(scores.size(0)))
        events = read_events(output)
        assert events[-1]['event'] == 'before_ce'
        assert events[-1]['shape'] == list(scores.shape)
        return forward(scores, targets)

    monkeypatch.setattr(diagnostic.functional, 'cross_entropy', observed_forward)
    args = tiny_args(diagnostic, output, '--item-sizes', '1', '5', '--seed', '7')
    assert diagnostic.run(args) == 0
    expected_shapes = [[3, 3], [3, 6], [1, 2], [5, 10]] * 2
    assert backward_shapes == expected_shapes
    events = read_events(output)
    before = [event for event in events if event['event'] == 'before_ce']
    assert [event['shape'] for event in before] == expected_shapes
    assert [event['call'] for event in before] == list(range(1, 9))
    assert all(event['dtype'] == 'torch.float32' for event in before)
    assert all(event['stride'] == [event['shape'][1], 1] for event in before)
    assert all(event['target_dtype'] == 'torch.int64' for event in before)
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'completed'
    assert result['completed_iterations'] == 2
    assert result['completed_calls'] == 8
    assert result['kind'] == 'synthetic_cross_entropy_diagnostic'
    assert events[-1]['status'] == 'completed'
    assert {entry.name for entry in output.iterdir()} == {'metadata.json', 'events.jsonl', 'result.json'}


def test_same_seed_has_deterministic_metadata_and_losses(diagnostic, tmp_path):
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        assert diagnostic.run(tiny_args(diagnostic, output, '--seed', '73', '--progress-every', '1')) == 0
    metadata = [json.loads((output / 'metadata.json').read_text()) for output in outputs]
    assert metadata[0] == metadata[1]
    assert metadata[0]['seed'] == 73
    assert metadata[0]['torch_version'] == torch.__version__
    assert metadata[0]['device'] == 'cpu'
    assert metadata[0]['device_uuid'] is None
    losses = [[event['loss'] for event in read_events(output) if event['event'] == 'progress'] for output in outputs]
    assert len(losses[0]) == 2
    assert losses[0] == losses[1]


def test_boundary_logging_is_sampled_not_per_call(diagnostic, tmp_path):
    output = tmp_path / 'sampled'
    args = tiny_args(diagnostic, output, '--iterations', '5', '--progress-every', '3')
    assert diagnostic.run(args) == 0
    before = [event for event in read_events(output) if event['event'] == 'before_ce']
    assert [event['iteration'] for event in before] == [1, 1, 3, 3, 5, 5]
    assert [event['call'] for event in before] == [1, 2, 5, 6, 9, 10]
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['boundary_recording'] == 'first, progress and final rounds; last recorded call may lag'


@pytest.mark.parametrize('existing_kind', ['directory', 'file', 'symlink'])
def test_output_never_overwrites_existing_paths(diagnostic, tmp_path, existing_kind):
    target = tmp_path / 'existing'
    if existing_kind == 'directory':
        target.mkdir()
        (target / 'keep').write_text('original')
    elif existing_kind == 'file':
        target.write_text('original')
    else:
        target.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        diagnostic.run(tiny_args(diagnostic, target))
    if existing_kind == 'directory':
        assert list(target.iterdir()) == [target / 'keep']
        assert (target / 'keep').read_text() == 'original'
    elif existing_kind == 'file':
        assert target.read_text() == 'original'
    else:
        assert target.is_symlink()
        assert not (tmp_path / 'missing').exists()


def test_missing_cuda_is_failure_not_cpu_fallback(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    output = tmp_path / 'no_cuda'
    args = diagnostic.parse_args(['--output-dir', str(output)])
    assert diagnostic.run(args) != 0
    events = read_events(output)
    assert events[-1]['status'] == 'error'
    assert 'CUDA' in events[-1]['error']
    assert not any(event['event'] == 'before_ce' for event in events)
    assert not (output / 'result.json').exists()


@pytest.mark.parametrize('uuid', [None, 'GPU-synthetic-test'])
def test_cuda_metadata_records_uuid_when_available_without_allocating(diagnostic, monkeypatch, uuid):
    properties = SimpleNamespace(name='synthetic device', major=7, minor=5, total_memory=4096)
    if uuid is not None:
        properties.uuid = uuid
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    queried = []

    def get_properties(index):
        queried.append(index)
        return properties

    monkeypatch.setattr(torch.cuda, 'get_device_properties', get_properties)
    assert diagnostic.device_metadata(torch.device('cuda')) == {
        'device': 'cuda:1',
        'device_name': 'synthetic device',
        'device_uuid': uuid,
        'device_capability': [7, 5],
        'device_total_memory': 4096,
    }
    assert diagnostic.device_metadata(torch.device('cuda:0'))['device'] == 'cuda:0'
    assert queried == [1, 0]


@pytest.mark.parametrize('expire_after_calls', [0, 1, 2])
def test_time_limit_is_incomplete_even_after_last_call(diagnostic, tmp_path, monkeypatch, expire_after_calls):
    output = tmp_path / 'timeout'
    clock = {'now': 0.0, 'calls': 0}
    forward = diagnostic.functional.cross_entropy
    original_metadata = diagnostic.device_metadata

    def metadata(device):
        result = original_metadata(device)
        if expire_after_calls == 0:
            clock['now'] = 2.0
        return result

    def expire_during_forward(scores, targets):
        result = forward(scores, targets)
        clock['calls'] += 1
        if clock['calls'] == expire_after_calls:
            clock['now'] = 2.0
        return result

    monkeypatch.setattr(diagnostic.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(diagnostic, 'device_metadata', metadata)
    monkeypatch.setattr(diagnostic.functional, 'cross_entropy', expire_during_forward)
    args = tiny_args(diagnostic, output, '--iterations', '1', '--max-seconds', '1')
    assert diagnostic.run(args) == 124
    assert not (output / 'result.json').exists()
    terminal = read_events(output)[-1]
    assert terminal['status'] == 'time_limit'
    assert terminal['completed_calls'] == expire_after_calls


@pytest.mark.parametrize('failure', ['loss', 'gradient', 'exception', 'interrupt'])
def test_failures_are_logged_and_never_success(diagnostic, tmp_path, monkeypatch, failure):
    output = tmp_path / failure
    forward = diagnostic.functional.cross_entropy

    def failing_forward(scores, targets):
        if failure == 'exception':
            raise RuntimeError('synthetic failure')
        if failure == 'interrupt':
            raise KeyboardInterrupt
        if failure == 'gradient':
            scores.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        loss = forward(scores, targets)
        return loss * float('nan') if failure == 'loss' else loss

    monkeypatch.setattr(diagnostic.functional, 'cross_entropy', failing_forward)
    assert diagnostic.run(tiny_args(diagnostic, output)) != 0
    assert not (output / 'result.json').exists()
    terminal = read_events(output)[-1]
    assert terminal['status'] == ('interrupted' if failure == 'interrupt' else 'error')
    assert terminal['completed_calls'] == 0
    assert terminal['completed_iterations'] == 0
    if failure in {'loss', 'gradient'}:
        assert 'nonfinite' in terminal['error']


def test_cli_cpu_smoke_and_existing_output_exit_status(tmp_path):
    assert SCRIPT.is_file(), 'Standalone cross-entropy diagnostic is not implemented'
    output = tmp_path / 'cli'
    command = [
        sys.executable,
        str(SCRIPT),
        '--output-dir',
        str(output),
        '--device',
        'cpu',
        '--batch-size',
        '2',
        '--iterations',
        '1',
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])['status'] == 'completed'
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert 'exist' in repeated.stderr.lower()
