import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_device_parity.py'
LOGQ_CASES = ('mclsr_q0_masked', 'mclsr_logq_il', 'mclsr_logq_uc', 'mclsr_logq_ic', 'mclsr_logq_all')
AUXILIARIES = ('L_IL', 'L_UC', 'L_IC')


def _verifier():
    assert SCRIPT.is_file(), 'The portable device-parity verifier is missing'
    spec = importlib.util.spec_from_file_location('check_device_parity', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli(tmp_path, *args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for variable in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'POLARS_MAX_THREADS',
    ):
        env[variable] = '1'
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cpu_self_check_cli(tmp_path):
    _verifier()
    report_path = tmp_path / 'parity.json'
    result = _cli(tmp_path, '--device', 'cpu', '--report', report_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report['passed'] is True
    assert report['reference_device'] == report['target_device'] == 'cpu'
    assert set(report['models']) == {'mclsr', 'sasrec'}
    for name, model_report in report['models'].items():
        checks = model_report['checks']
        assert checks and all(check['passed'] for check in checks)
        assert all(check['finite'] for check in checks)
        assert all(check['max_abs_error'] == 0.0 for check in checks)
        assert any(check['name'].startswith('gradient/') for check in checks)
        loss_names = {check['name'] for check in checks if check['name'].startswith('loss/')}
        expected = {'loss/L_P', 'loss/total'}
        if name == 'mclsr':
            expected |= {'loss/L_IL', 'loss/L_UC', 'loss/L_IC'}
        assert loss_names == expected


def test_injected_mismatch_fails_cli(tmp_path, monkeypatch):
    verifier = _verifier()
    original = verifier.run_once
    calls = 0

    def mismatched(*args, **kwargs):
        nonlocal calls
        snapshot = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            snapshot['output/combined_representation'][0, 0] += 1.0
        return snapshot

    monkeypatch.setattr(verifier, 'run_once', mismatched)
    report_path = tmp_path / 'mismatch.json'
    status = verifier.main(['--device', 'cpu', '--report', str(report_path)])
    assert status == 1
    report = json.loads(report_path.read_text())
    assert report['passed'] is False
    failed = [check for check in report['models']['mclsr']['checks'] if not check['passed']]
    assert [check['name'] for check in failed] == ['output/combined_representation']
    assert failed[0]['max_abs_error'] > 0.9


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
@pytest.mark.parametrize('side', ['reference', 'target'])
def test_comparison_rejects_nonfinite(bad_value, side):
    verifier = _verifier()
    values = {'reference': torch.tensor([1.0]), 'target': torch.tensor([1.0])}
    values[side][0] = bad_value
    result = verifier.compare_tensors(values['reference'], values['target'], atol=1e-4, rtol=1e-3)
    assert result['passed'] is False
    assert result['finite'] is False
    json.dumps(result, allow_nan=False)


def test_comparison_rejects_shape_and_missing_gradient():
    verifier = _verifier()
    for target in (torch.ones(2), None):
        result = verifier.compare_tensors(torch.ones(1), target, atol=1e-4, rtol=1e-3)
        assert result['passed'] is False
    assert not verifier.compare_tensors(None, None, atol=1e-4, rtol=1e-3)['passed']


def test_comparison_uses_elementwise_absolute_plus_relative_tolerance():
    verifier = _verifier()
    reference = torch.tensor([0.0, 100.0], dtype=torch.float64)
    target = torch.tensor([5e-5, 100.05], dtype=torch.float64)
    result = verifier.compare_tensors(reference, target, atol=1e-4, rtol=1e-3)
    assert result['passed'] is True
    assert result['max_abs_error'] == pytest.approx(0.05)
    assert result['max_rel_error'] > 0
    target[0] = 0.01
    assert not verifier.compare_tensors(reference, target, atol=1e-4, rtol=1e-3)['passed']
    assert not verifier.compare_tensors(torch.tensor([1]), torch.tensor([2]), atol=10, rtol=10)['passed']


def test_unavailable_cuda_is_an_error_not_a_skip(tmp_path):
    _verifier()
    report_path = tmp_path / 'unavailable.json'
    result = _cli(tmp_path, '--device', 'cuda:0', '--report', report_path)
    assert result.returncode != 0
    assert 'CUDA is unavailable' in result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report['passed'] is False
    assert 'CUDA is unavailable' in report['error']


def test_device_scope_restores_legacy_globals_on_error(monkeypatch):
    verifier = _verifier()
    original = torch.device('cuda:7')
    monkeypatch.setattr(verifier.utils, 'DEVICE', original)
    monkeypatch.setattr(verifier.model_base, 'DEVICE', original)
    with pytest.raises(RuntimeError, match='injected'):
        with verifier.device_scope(torch.device('cpu')):
            assert verifier.utils.DEVICE == torch.device('cpu')
            assert verifier.model_base.DEVICE == torch.device('cpu')
            raise RuntimeError('injected')
    assert verifier.utils.DEVICE == verifier.model_base.DEVICE == original


def test_logq_suite_cpu_cli_and_provenance(tmp_path):
    result = _cli(tmp_path, '--device', 'cpu', '--suite', 'logq', '--report', tmp_path / 'logq.json')
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'logq.json').read_text())
    assert report['suite'] == 'logq'
    assert report['passed'] is True
    assert report['atol'] == 1e-4 and report['rtol'] == 1e-3
    assert tuple(report['models']) == LOGQ_CASES
    for result in report['models'].values():
        assert len(result['checks']) == 40
        assert all(check['passed'] and check['finite'] and check['max_abs_error'] == 0 for check in result['checks'])
        assert set(result['components']) == {'L_P', *AUXILIARIES, 'total'}
        assert all(value['reference'] == value['target'] for value in result['components'].values())
        provenance = result['provenance']
        encoded = json.dumps(provenance['config'], sort_keys=True, allow_nan=False).encode()
        assert provenance['config_sha256'] == hashlib.sha256(encoded).hexdigest()
        for source, digest in provenance['source_sha256'].items():
            assert hashlib.sha256((SCRIPT.parents[1] / source).read_bytes()).hexdigest() == digest
        assert 'src/irec/loss/contrastive.py' in provenance['source_sha256']
        assert 'src/irec/models/mclsr.py' in provenance['source_sha256']


def test_logq_variants_are_matched_and_use_production_role_aware_losses(tmp_path, monkeypatch):
    verifier = _verifier()
    from irec.loss import FpsLogQLoss, InBatchLogQLoss

    original = verifier.run_once
    captured = []

    def capture(model, losses, weights, batch, device):
        assert isinstance(losses['L_P'], InBatchLogQLoss)
        assert losses['L_P']._logq_lambda == 1.0
        assert all(isinstance(losses[name], FpsLogQLoss) for name in AUXILIARIES)
        assert batch['user.ids'].unique().numel() < batch['user.ids'].numel()
        assert batch['item.ids'].unique().numel() < batch['item.ids'].numel()
        assert all(module.p == 0 for module in model.modules() if isinstance(module, torch.nn.Dropout))
        snapshot = original(model, losses, weights, batch, device)
        assert {key.removeprefix('gradient/') for key in snapshot if key.startswith('gradient/')} == {
            name for name, _ in model.named_parameters()
        }
        assert all(value is not None for value in snapshot.values())
        captured.append(snapshot)
        return snapshot

    monkeypatch.setattr(verifier, 'run_once', capture)
    report = verifier.run_parity('cpu', suite='logq')
    assert report['passed']
    assert len(captured) == 2 * len(LOGQ_CASES)
    control = report['models'][LOGQ_CASES[0]]
    for index, (name, enabled) in enumerate(zip(LOGQ_CASES, ((), ('L_IL',), ('L_UC',), ('L_IC',), AUXILIARIES))):
        result = report['models'][name]
        config = result['provenance']['config']
        normalized = copy.deepcopy(config)
        for component in AUXILIARIES:
            loss_config = config['losses'][component]['kwargs']
            assert loss_config['logq_lambda'] == float(component in enabled)
            assert loss_config['mask_false_negatives'] is True
            assert loss_config['leave_own_out'] is False
            assert loss_config['center_log_q'] is False
            assert loss_config['scheme'] == 'paper'
            assert loss_config['tau'] == 0.5
            assert loss_config['normalize_embeddings'] is True
            is_item = component == 'L_IC'
            assert loss_config['ids_prefix'] == ('item_ids' if is_item else 'user_ids')
            assert loss_config['expected_counts_role'] == ('context-inclusion' if is_item else 'user-row')
            assert loss_config['logq_probability_mode'] == ('inbatch_negative' if is_item else 'sample')
            assert loss_config['num_draws_value'] == (4 if is_item else None)
            artifact = config['count_artifacts'][loss_config['path_to_counts']]
            assert artifact['role'] == loss_config['expected_counts_role']
            assert artifact['max_len'] == loss_config['expected_counts_max_len'] == 5
            normalized['losses'][component]['kwargs']['logq_lambda'] = 0.0
            baseline_value = control['components'][component]['reference']
            value = result['components'][component]['reference']
            assert (value != baseline_value) == (component in enabled)
        assert normalized == control['provenance']['config']
        for key in captured[0]:
            if key.startswith('output/'):
                assert torch.equal(captured[0][key], captured[2 * index][key])
        if enabled:
            assert any(
                not torch.equal(captured[0][key], captured[2 * index][key])
                for key in captured[0]
                if key.startswith('gradient/')
            )
        components = result['components']
        assert components['L_P'] == control['components']['L_P']
        assert components['total']['reference'] == pytest.approx(
            sum(config['weights'][component] * components[component]['reference'] for component in config['weights'])
        )


@pytest.mark.parametrize(
    'case_index,key,corruption',
    [
        (0, 'output/combined_representation', 'mismatch'),
        (1, 'loss/L_IL', 'mismatch'),
        (2, 'gradient/_user_embeddings.weight', 'mismatch'),
        (3, 'gradient/_item_embeddings.weight', 'missing'),
        (4, 'loss/L_IC', 'nonfinite'),
    ],
)
def test_logq_injected_failures_are_not_skipped(tmp_path, monkeypatch, case_index, key, corruption):
    verifier = _verifier()
    original = verifier.run_once
    calls = 0

    def corrupted(*args):
        nonlocal calls
        snapshot = original(*args)
        calls += 1
        if calls == 2 * case_index + 2:
            if corruption == 'missing':
                snapshot.pop(key)
            elif corruption == 'nonfinite':
                snapshot[key].fill_(float('nan'))
            else:
                snapshot[key] += 1.0
        return snapshot

    monkeypatch.setattr(verifier, 'run_once', corrupted)
    path = tmp_path / 'failed-logq.json'
    assert verifier.main(['--device', 'cpu', '--suite', 'logq', '--report', str(path)]) == 1
    report = json.loads(path.read_text())
    assert not report['passed']
    failed = report['models'][LOGQ_CASES[case_index]]
    assert [check['name'] for check in failed['checks'] if not check['passed']] == [key]
    assert calls == 10


def test_explicit_base_matches_default_and_retains_unmasked_losses():
    verifier = _verifier()
    default = verifier.run_parity('cpu')
    explicit = verifier.run_parity('cpu', suite='base')
    assert default == explicit
    assert tuple(default['models']) == ('mclsr', 'sasrec')
    config = default['models']['mclsr']['provenance']['config']
    assert config['weights'] == {'L_P': 1.0, 'L_IL': 1.0, 'L_UC': 0.05, 'L_IC': 0.05}
    for component in AUXILIARIES:
        assert config['losses'][component]['class'] == 'irec.loss.contrastive.FpsLoss'
    assert default['models']['mclsr']['components']['total']['reference'] == pytest.approx(5.032530307769775)
    assert default['models']['sasrec']['components']['total']['reference'] == pytest.approx(5.154390811920166)
