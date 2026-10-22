import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def trace_module():
    assert importlib.util.find_spec('trace_cross_entropy') is not None, 'CE trace module is missing'
    return importlib.import_module('trace_cross_entropy')


def read_trace(path):
    return json.loads(path.read_bytes().rstrip(b'\0 \n'))


def test_trace_preserves_loss_and_gradients(tmp_path):
    scores = torch.tensor([[0.2, 0.7], [0.8, 0.1]], requires_grad=True)
    labels = torch.arange(2)
    expected = functional.cross_entropy(scores, labels)
    expected_grad = torch.autograd.grad(expected, scores)[0]
    original = functional.cross_entropy
    path = tmp_path / 'trace.json'
    with trace_module().trace_cross_entropy(path):
        actual = functional.cross_entropy(scores, labels)
        gradient = torch.autograd.grad(actual, scores)[0]
        recorded = read_trace(path)
    assert functional.cross_entropy is original
    assert torch.equal(actual, expected)
    assert torch.equal(gradient, expected_grad)
    assert recorded['phase'] == 'returned'
    assert recorded['call'] == 1
    assert recorded['input']['shape'] == [2, 2]
    assert recorded['input']['stride'] == [2, 1]
    assert recorded['input']['dtype'] == 'torch.float32'
    assert recorded['target']['dtype'] == 'torch.int64'
    assert recorded['weight'] is None


def test_pre_call_trace_visible_to_callee_and_error_propagates(tmp_path, monkeypatch):
    path = tmp_path / 'trace.json'

    def failing_loss(*args, **kwargs):
        assert read_trace(path)['phase'] == 'enter'
        raise RuntimeError('deliberate failure')

    monkeypatch.setattr(functional, 'cross_entropy', failing_loss)
    with pytest.raises(RuntimeError, match='deliberate failure'):
        with trace_module().trace_cross_entropy(path):
            functional.cross_entropy(torch.zeros(2, 2), torch.arange(2))
    assert functional.cross_entropy is failing_loss
    assert read_trace(path)['phase'] == 'enter'


def test_trace_captures_module_loss_identity_and_kwargs(tmp_path):
    class NamedLoss:
        _output_prefix = 'train/interest'

        def forward(self, scores, labels):
            return functional.cross_entropy(input=scores, target=labels, reduction='sum')

    path = tmp_path / 'trace.json'
    with trace_module().trace_cross_entropy(path):
        NamedLoss().forward(torch.ones(3, 3), torch.arange(3))
    recorded = read_trace(path)
    assert recorded['loss'] == 'train/interest'
    assert recorded['reduction'] == 'sum'
    assert any(frame['function'] == 'forward' for frame in recorded['python_stack'])


def test_trace_is_bounded_and_refuses_overwrite(tmp_path):
    module = trace_module()
    path = tmp_path / 'trace.json'
    with module.trace_cross_entropy(path):
        for _ in range(10):
            functional.cross_entropy(torch.ones(2, 2), torch.arange(2))
    assert path.stat().st_size == module.TRACE_BYTES
    assert read_trace(path)['call'] == 10
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        with module.trace_cross_entropy(path):
            pytest.fail('must not enter')
    assert path.read_bytes() == before


def test_cli_restores_argv_and_does_not_swallow_target_failure(tmp_path, monkeypatch):
    module = trace_module()
    target = tmp_path / 'target.py'
    target.write_text('import sys\nassert sys.argv[1:] == ["--seed", "3"]\nraise SystemExit(17)\n')
    arguments = ['trace_cross_entropy.py', '--trace-file', str(tmp_path / 'trace.json'), str(target), '--seed', '3']
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 17
    assert sys.argv is arguments


def test_real_training_trace_covers_four_losses_without_test(tmp_path):
    from test_end_to_end import _config, _synthetic_dataset

    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / 'data'
    num_items = _synthetic_dataset(str(data_root / 'Tiny'))
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(_config(str(data_root), 'Tiny', num_items, steps=8)))
    report = tmp_path / 'unused_test_report.json'
    trace = tmp_path / 'trace.json'
    code = '\n'.join(
        [
            'import json',
            'import torch.nn.functional as functional',
            'import train_confirmatory as runner',
            'from diagnose_training import run_training_only',
            'from trace_cross_entropy import trace_cross_entropy',
            'from pathlib import Path',
            f'trace_path = Path({str(trace)!r})',
            'original = functional.cross_entropy',
            'states = []',
            'def observed(*args, **kwargs):',
            '    states.append(json.loads(trace_path.read_bytes().rstrip(b"\\0 \\n")))',
            '    return original(*args, **kwargs)',
            'functional.cross_entropy = observed',
            'with trace_cross_entropy(trace_path):',
            f'    result = run_training_only(runner, ["--params", {str(params)!r}, "--seed", "1", "--select", "validation/ndcg@20", "--output", {str(report)!r}])',
            'print("TRACE_RESULT", json.dumps({"result": result, "states": states}))',
        ]
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=os.pathsep.join([str(root / 'src'), str(root / 'scripts')]), CUDA_VISIBLE_DEVICES=''
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(
        next(
            line.removeprefix('TRACE_RESULT ')
            for line in result.stdout.splitlines()
            if line.startswith('TRACE_RESULT ')
        )
    )
    assert payload['result']['optimizer_updates'] == 8
    assert not payload['result']['test_evaluated']
    assert not report.exists()
    assert len(payload['states']) == 32
    assert len({state['loss'] for state in payload['states']}) == 4
    assert all(state['loss'] and state['phase'] == 'enter' for state in payload['states'])
    assert [state['optimizer_step_zero_based'] for state in payload['states']] == [
        step for step in range(8) for _ in range(4)
    ]
