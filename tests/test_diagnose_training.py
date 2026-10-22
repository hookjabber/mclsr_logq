import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def diagnostic_module():
    assert importlib.util.find_spec('diagnose_training') is not None, 'diagnostic runner is missing'
    return importlib.import_module('diagnose_training')


def fake_runner(updates=3, failure=None):
    calls = []

    def train(**kwargs):
        for step in range(updates):
            kwargs['callback']({}, step)
        if failure:
            raise failure
        return {'validation/ndcg@20': {'step': 1, 'value': 0.2}}

    def evaluate():
        calls.append('test')

    runner = SimpleNamespace(train=train, evaluate=evaluate)

    def main():
        runner.train(callback=lambda batch, step: calls.append(step), step_cnt=3)
        calls.append('test_loader')
        runner.evaluate()

    runner.main = main
    return runner, calls


def test_stops_before_test_and_restores_original_runner():
    runner, calls = fake_runner()
    original_train, original_evaluate, original_argv = runner.train, runner.evaluate, sys.argv
    result = diagnostic_module().run_training_only(runner, ['--params', 'diagnostic.json'])
    assert calls == [0, 1, 2]
    assert result == {'status': 'training_finished', 'optimizer_updates': 3, 'test_evaluated': False}
    assert runner.train is original_train
    assert runner.evaluate is original_evaluate
    assert sys.argv is original_argv


def test_propagates_training_failure_without_evaluation():
    runner, calls = fake_runner(failure=RuntimeError('CUDA failure'))
    original_train = runner.train
    with pytest.raises(RuntimeError, match='CUDA failure'):
        diagnostic_module().run_training_only(runner, [])
    assert 'test' not in calls and 'test_loader' not in calls
    assert runner.train is original_train


def test_rejects_early_training_return():
    runner, calls = fake_runner(updates=2)
    with pytest.raises(RuntimeError, match='update budget'):
        diagnostic_module().run_training_only(runner, [])
    assert 'test_loader' not in calls


def test_blocks_unexpected_evaluation_before_training():
    runner, calls = fake_runner()
    runner.main = lambda: runner.evaluate()
    with pytest.raises(RuntimeError, match='test evaluation is forbidden'):
        diagnostic_module().run_training_only(runner, [])
    assert calls == []


def test_rejects_runner_returning_without_training():
    runner, _ = fake_runner()
    runner.main = lambda: None
    with pytest.raises(RuntimeError, match='without training'):
        diagnostic_module().run_training_only(runner, [])


def test_real_runner_exits_before_test_report(tmp_path):
    from test_end_to_end import _config, _synthetic_dataset

    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / 'data'
    num_items = _synthetic_dataset(str(data_root / 'Tiny'))
    config = _config(str(data_root), 'Tiny', num_items, steps=16)
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(config))
    report = tmp_path / 'unused_test_report.json'
    arguments = ['--params', str(params), '--seed', '1', '--select', 'validation/ndcg@20', '--output', str(report)]
    code = (
        'import json; import train_confirmatory as runner; '
        'from diagnose_training import run_training_only; '
        f'print("DIAGNOSTIC_RESULT", json.dumps(run_training_only(runner, {arguments!r})))'
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
    result_line = next(line for line in result.stdout.splitlines() if line.startswith('DIAGNOSTIC_RESULT '))
    assert json.loads(result_line.split(' ', 1)[1]) == {
        'status': 'training_finished',
        'optimizer_updates': 16,
        'test_evaluated': False,
    }
    assert not report.exists()
    assert Path(str(report) + '.provenance').exists()
    assert list((tmp_path / 'checkpoints').glob('*.pth'))
