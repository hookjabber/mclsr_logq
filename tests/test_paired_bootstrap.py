"""The paired statistics behind results/stats: interval coverage and the sign-flip p-value."""

import json
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paired_bootstrap import paired_stats  # noqa: E402
import paired_bootstrap


def test_clear_effect_and_null():
    rng = np.random.default_rng(0)
    users, seeds = 2000, 3
    effect = 0.02 + rng.normal(0, 0.05, size=(users, seeds))  # per-user differences around +0.02
    obs, lo, hi, p = paired_stats(effect, np.random.default_rng(1), draws=2000)
    assert abs(obs - effect.mean()) < 1e-12
    assert lo < 0.02 < hi and hi - lo < 0.01 and p < 0.001
    null = rng.normal(0, 0.05, size=(users, seeds))
    obs, lo, hi, p = paired_stats(null, np.random.default_rng(1), draws=2000)
    assert lo < 0 < hi and p > 0.05


def test_interval_tracks_sample_size():
    rng = np.random.default_rng(0)
    wide = paired_stats(rng.normal(0.01, 0.1, size=(200, 3)), np.random.default_rng(1), draws=2000)
    narrow = paired_stats(rng.normal(0.01, 0.1, size=(5000, 3)), np.random.default_rng(1), draws=2000)
    assert (wide[2] - wide[1]) > 3 * (narrow[2] - narrow[1])


def _scores(user_ids, values):
    return {
        'user_ids': np.asarray(user_ids),
        **{name: np.asarray(values, dtype=np.float64) for name in paired_bootstrap.METRICS},
    }


def _configure_main(monkeypatch, tmp_path, evaluations, seeds=(1, 2, 3), draws=101):
    output = tmp_path / 'comparison.md'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'paired_bootstrap.py',
            '--prefix',
            'fixture',
            '--config-dir',
            str(tmp_path),
            '--pairs',
            'arm:base',
            '--seeds',
            *map(str, seeds),
            '--draws',
            str(draws),
            '--output',
            str(output),
        ],
    )
    monkeypatch.setattr(
        paired_bootstrap,
        'load_arm',
        lambda prefix, arm, seed, config_dir, confirm_dir, cache: evaluations[(arm, seed)],
    )
    return output


def test_main_aligns_user_ids_across_arms_and_seeds(monkeypatch, tmp_path):
    evaluations = {
        ('arm', 1): _scores([30, 10, 20], [0.8, 0.2, 0.5]),
        ('base', 1): _scores([20, 30, 10], [0.3, 0.5, 0.1]),
        ('arm', 2): _scores([20, 10, 30], [0.3, 0.1, 0.5]),
        ('base', 2): _scores([10, 30, 20], [0.2, 0.8, 0.5]),
        ('arm', 3): _scores([10, 20, 30], [0.4, 0.6, 0.8]),
        ('base', 3): _scores([30, 20, 10], [0.5, 0.3, 0.1]),
    }
    output = _configure_main(monkeypatch, tmp_path, evaluations)
    paired_bootstrap.main()
    report = json.loads(output.with_suffix('.json').read_text())
    expected = np.array([[0.1, -0.1, 0.3], [0.2, -0.2, 0.3], [0.3, -0.3, 0.3]])
    for metric in paired_bootstrap.METRICS:
        result = report[f'arm:base:{metric}']
        assert result['user_ids'] == [10, 20, 30]
        np.testing.assert_allclose(result['per_user_differences'], expected)
        np.testing.assert_allclose(result['per_seed'], expected.mean(axis=0))
        assert result['mean'] == pytest.approx(expected.mean())


@pytest.mark.parametrize('arm,seed', [('arm', 1), ('base', 1), ('arm', 2), ('base', 3)])
@pytest.mark.parametrize('defect', ['missing_ids', 'duplicate_ids', 'different_ids', 'metric_length', 'nonfinite'])
def test_main_rejects_invalid_pairing(monkeypatch, tmp_path, arm, seed, defect):
    evaluations = {(name, run): _scores([10, 20], [0.2, 0.4]) for name in ('arm', 'base') for run in (1, 2, 3)}
    invalid = evaluations[(arm, seed)]
    if defect == 'missing_ids':
        del invalid['user_ids']
    elif defect == 'duplicate_ids':
        invalid['user_ids'] = np.array([10, 10])
    elif defect == 'different_ids':
        invalid['user_ids'] = np.array([10, 30])
    elif defect == 'metric_length':
        invalid['ndcg@20'] = np.array([0.2])
    else:
        invalid['ndcg@20'][0] = np.nan
    output = _configure_main(monkeypatch, tmp_path, evaluations)
    with pytest.raises(ValueError, match='user|finite'):
        paired_bootstrap.main()
    assert not output.exists()
    assert not output.with_suffix('.json').exists()


def test_report_is_explicitly_conditional_on_trained_seeds(monkeypatch, tmp_path, capsys):
    user_ids = np.arange(100)
    evaluations = {}
    for seed, difference in zip((1, 2, 3), (-0.05, 0.10, 0.10)):
        evaluations[('base', seed)] = _scores(user_ids, np.full(100, 0.2))
        evaluations[('arm', seed)] = _scores(user_ids, np.full(100, 0.2 + difference))
    output = _configure_main(monkeypatch, tmp_path, evaluations, draws=101)
    paired_bootstrap.main()
    report = json.loads(output.with_suffix('.json').read_text())
    for metric in paired_bootstrap.METRICS:
        result = report[f'arm:base:{metric}']
        np.testing.assert_allclose(result['per_seed'], [-0.05, 0.10, 0.10])
        np.testing.assert_allclose(result['ci95'], [0.05, 0.05])
        assert result['inference_scope'] == 'conditional_on_trained_seeds'
        assert result['resampling_unit'] == 'user'
        assert result['training_seed_uncertainty'] == 'not_estimated'
        assert result['per_seed_inference'] == 'not_performed'
        assert result['equivalence_test'] == 'not_performed'
        assert result['multiple_testing_adjustment'] == 'none'
        assert result['checkpoint_selection_metric'] == 'validation/ndcg@20'
        assert result['draws'] == 101
        assert result['per_seed_sd'] == pytest.approx(np.std([-0.05, 0.10, 0.10], ddof=1))
    markdown = output.read_text()
    assert 'conditional on the trained seeds' in markdown
    assert 'not per-seed significance' in markdown
    assert 'not evidence of equivalence' in markdown
    assert '101 draws' in markdown
    assert '10 000 draws' not in markdown
    assert 'seed 1: -0.0500' in markdown
    assert 'validation/ndcg@20' in markdown
    assert 'conditional on the trained seeds' in capsys.readouterr().out


@pytest.mark.parametrize('suffix', ['.md', '.json'])
def test_main_preserves_existing_reports(monkeypatch, tmp_path, suffix):
    output = _configure_main(monkeypatch, tmp_path, {})
    existing = output.with_suffix(suffix)
    existing.write_text('historical report\n')
    with pytest.raises(FileExistsError, match='exist'):
        paired_bootstrap.main()
    assert existing.read_text() == 'historical report\n'
    other = output.with_suffix('.json' if suffix == '.md' else '.md')
    assert not other.exists()


def test_main_rejects_duplicate_training_seeds(monkeypatch, tmp_path):
    _configure_main(monkeypatch, tmp_path, {}, seeds=(1, 1))
    with pytest.raises(ValueError, match='seeds.*unique'):
        paired_bootstrap.main()


@pytest.mark.parametrize('user_prefix', ['user', 'account'])
def test_per_user_metrics_keeps_observed_batch_ids(monkeypatch, user_prefix):
    batches = [
        {
            f'{user_prefix}.ids': torch.tensor([9, 2]),
            'labels.ids': torch.tensor([1, 1500]),
            'labels.length': torch.tensor([1, 1]),
        },
        {
            f'{user_prefix}.ids': torch.tensor([7]),
            'labels.ids': torch.tensor([2]),
            'labels.length': torch.tensor([1]),
        },
    ]

    class RankingModel(torch.nn.Module):
        def forward(self, batch):
            return torch.arange(1, 1001).expand(batch[f'{user_prefix}.ids'].numel(), -1)

    monkeypatch.setattr(paired_bootstrap, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(paired_bootstrap, 'dataset_for', lambda config: (SimpleNamespace(meta={}), batches))
    monkeypatch.setattr(paired_bootstrap.BaseModel, 'create_from_config', lambda config, **meta: RankingModel())
    monkeypatch.setattr(paired_bootstrap.torch, 'load', lambda *args, **kwargs: {})
    config = {
        'model': {'user_prefix': user_prefix},
        'callback': {'callbacks': [{'type': 'eval', 'pred_prefix': 'predictions', 'labels_prefix': 'labels'}]},
    }
    result = paired_bootstrap.per_user_metrics(config, 'fixture.pth')
    np.testing.assert_array_equal(result['user_ids'], [9, 2, 7])
    np.testing.assert_allclose(result['ndcg@20'], [1, 0, 1 / np.log2(3)])
    np.testing.assert_allclose(result['recall@1000'], [1, 0, 1])


@pytest.mark.parametrize(
    'differences',
    [np.zeros((0, 2)), np.zeros((2, 0)), np.zeros(2), np.full((2, 2), np.nan), np.full((2, 2), np.inf)],
)
def test_paired_stats_rejects_invalid_differences(differences):
    with pytest.raises(ValueError, match='differences'):
        paired_stats(differences, np.random.default_rng(0), draws=10)


@pytest.mark.parametrize('draws', [0, -1, 1.5, True])
def test_paired_stats_requires_positive_integer_draws(draws):
    with pytest.raises(ValueError, match='draws'):
        paired_stats(np.zeros((2, 2)), np.random.default_rng(0), draws=draws)


@pytest.fixture
def arm_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / 'configs/arm.json'
    checkpoint = tmp_path / 'checkpoints/model.pth'
    report_path = tmp_path / 'results/confirm/tiny_arm_seed1.json'
    for path in (config_path, checkpoint, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        'model': {},
        'dataset': {'type': 'graph', 'dataset': {'type': 'mclsr', 'path_to_data_dir': './data', 'name': 'Tiny'}},
        'callback': {'callbacks': [{'type': 'eval', 'pred_prefix': 'predictions', 'labels_prefix': 'labels'}]},
    }
    config_path.write_text(json.dumps(config))
    checkpoint.write_bytes(b'checkpoint fixture')
    report = {
        'seed': 1,
        'params': '/original/repository/configs/arm.json',
        'params_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'results': {
            'validation/ndcg@20': {
                'selection_metric': 'validation/ndcg@20',
                'checkpoint': './checkpoints/model.pth',
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        },
    }
    report_path.write_text(json.dumps(report))
    calls = []
    scores = _scores([2, 9], [0.2, 0.3])

    def score(config, path):
        calls.append((config, Path(path).resolve()))
        return scores

    monkeypatch.setattr(paired_bootstrap, 'per_user_metrics', score)
    return SimpleNamespace(
        root=tmp_path,
        config_path=config_path,
        checkpoint=checkpoint,
        report_path=report_path,
        config=config,
        report=report,
        calls=calls,
        scores=scores,
        cache={},
    )


def _load_fixture_arm(arm_files):
    return paired_bootstrap.load_arm(
        'tiny', 'arm', 1, str(arm_files.config_path.parent), str(arm_files.report_path.parent), arm_files.cache
    )


def _add_provenance(arm_files):
    root = arm_files.root / 'data/Tiny'
    root.mkdir(parents=True)
    inputs = {}
    for name in (
        'train_mclsr.txt',
        'train_sasrec.txt',
        'valid_history.txt',
        'valid_target.txt',
        'test_history.txt',
        'test_target.txt',
    ):
        path = root / name
        path.write_text('2 3 4\n')
        inputs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        'schema_version': 1,
        'seed': 1,
        'working_directory': str(arm_files.root),
        'params': {'path': str(arm_files.config_path), 'sha256': arm_files.report['params_sha256']},
        'evaluation_config': arm_files.config['callback']['callbacks'][0],
        'datasets': [{'root': str(root), 'inputs': inputs}],
    }
    _save_provenance(arm_files, manifest)
    return manifest


def _save_provenance(arm_files, manifest):
    path = Path(str(arm_files.report_path) + '.provenance')
    path.write_text(json.dumps(manifest))
    arm_files.report['provenance'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    arm_files.report_path.write_text(json.dumps(arm_files.report))


@pytest.mark.parametrize('artifact', ['config', 'checkpoint'])
def test_load_arm_rejects_changed_artifact_bytes_before_scoring(arm_files, artifact):
    path = arm_files.config_path if artifact == 'config' else arm_files.checkpoint
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA-256'):
        _load_fixture_arm(arm_files)
    assert not arm_files.calls
    assert not arm_files.cache


@pytest.mark.parametrize('artifact', ['config', 'checkpoint'])
@pytest.mark.parametrize('invalid_hash', [None, '', 'not-a-hash'])
def test_load_arm_requires_valid_artifact_hashes(arm_files, artifact, invalid_hash):
    record = arm_files.report if artifact == 'config' else arm_files.report['results']['validation/ndcg@20']
    field = 'params_sha256' if artifact == 'config' else 'checkpoint_sha256'
    if invalid_hash is None:
        del record[field]
    else:
        record[field] = invalid_hash
    arm_files.report_path.write_text(json.dumps(arm_files.report))
    with pytest.raises(ValueError, match='SHA-256'):
        _load_fixture_arm(arm_files)
    assert not arm_files.calls


@pytest.mark.parametrize('absolute_checkpoint', [False, True])
def test_load_arm_accepts_verified_legacy_pair_and_preserves_files(arm_files, absolute_checkpoint):
    if absolute_checkpoint:
        arm_files.report['results']['validation/ndcg@20']['checkpoint'] = str(arm_files.checkpoint)
        arm_files.report_path.write_text(json.dumps(arm_files.report))
    paths = (arm_files.config_path, arm_files.checkpoint, arm_files.report_path)
    before = [path.read_bytes() for path in paths]
    assert _load_fixture_arm(arm_files) is arm_files.scores
    assert _load_fixture_arm(arm_files) is arm_files.scores
    assert arm_files.calls == [(arm_files.config, arm_files.checkpoint)]
    assert [path.read_bytes() for path in paths] == before


def test_load_arm_does_not_bypass_verification_on_cache_hit(arm_files):
    _load_fixture_arm(arm_files)
    arm_files.checkpoint.write_bytes(b'replaced after scoring')
    with pytest.raises(ValueError, match='SHA-256'):
        _load_fixture_arm(arm_files)
    assert len(arm_files.calls) == 1


@pytest.mark.parametrize('defect', ['seed', 'selection_metric'])
def test_load_arm_rejects_wrong_run_or_selection(arm_files, defect):
    if defect == 'seed':
        arm_files.report['seed'] = 2
    else:
        arm_files.report['results']['validation/ndcg@20']['selection_metric'] = 'validation/recall@1000'
    arm_files.report_path.write_text(json.dumps(arm_files.report))
    with pytest.raises(ValueError, match='seed|selection'):
        _load_fixture_arm(arm_files)
    assert not arm_files.calls


def test_load_arm_accepts_verified_provenance_and_refreshes_dataset_cache(arm_files, monkeypatch):
    _add_provenance(arm_files)
    key = json.dumps(arm_files.config['dataset'], sort_keys=True)
    monkeypatch.setitem(paired_bootstrap._DATASETS, key, 'stale loader')
    assert _load_fixture_arm(arm_files) is arm_files.scores
    assert key not in paired_bootstrap._DATASETS


@pytest.mark.parametrize('artifact', ['sidecar', 'test_history.txt', 'test_target.txt', 'train_mclsr.txt'])
def test_load_arm_rejects_changed_provenance_inputs_before_scoring(arm_files, artifact):
    _add_provenance(arm_files)
    path = (
        Path(arm_files.report['provenance']['path'])
        if artifact == 'sidecar'
        else arm_files.root / 'data/Tiny' / artifact
    )
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA-256'):
        _load_fixture_arm(arm_files)
    assert not arm_files.calls


@pytest.mark.parametrize(
    'defect',
    ['sidecar_hash', 'datasets', 'history_record', 'target_hash', 'params_hash', 'evaluation_config', 'input_path'],
)
def test_load_arm_rejects_incomplete_or_unrelated_provenance(arm_files, defect):
    manifest = _add_provenance(arm_files)
    if defect == 'datasets':
        manifest['datasets'] = []
    elif defect == 'history_record':
        del manifest['datasets'][0]['inputs']['test_history.txt']
    elif defect == 'target_hash':
        del manifest['datasets'][0]['inputs']['test_target.txt']['sha256']
    elif defect == 'params_hash':
        manifest['params']['sha256'] = '0' * 64
    elif defect == 'evaluation_config':
        manifest['evaluation_config']['labels_prefix'] = 'other_labels'
    elif defect == 'input_path':
        manifest['datasets'][0]['inputs']['test_history.txt']['path'] = str(arm_files.root / 'unrelated.txt')
    _save_provenance(arm_files, manifest)
    if defect == 'sidecar_hash':
        del arm_files.report['provenance']['sha256']
        arm_files.report_path.write_text(json.dumps(arm_files.report))
    with pytest.raises(ValueError, match='provenance|SHA-256'):
        _load_fixture_arm(arm_files)
    assert not arm_files.calls


@pytest.mark.parametrize('change_inputs', [False, True])
def test_load_arm_checks_actual_relocated_inputs_not_only_recorded_paths(arm_files, monkeypatch, change_inputs):
    _add_provenance(arm_files)
    relocated = arm_files.root / 'relocated'
    data = relocated / 'data/Tiny'
    data.mkdir(parents=True)
    for original in (arm_files.root / 'data/Tiny').iterdir():
        (data / original.name).write_bytes(original.read_bytes())
    if change_inputs:
        (data / 'test_target.txt').write_text('2 8 9\n')
    arm_files.report['results']['validation/ndcg@20']['checkpoint'] = str(arm_files.checkpoint)
    arm_files.report_path.write_text(json.dumps(arm_files.report))
    monkeypatch.chdir(relocated)
    if change_inputs:
        with pytest.raises(ValueError, match='SHA-256'):
            _load_fixture_arm(arm_files)
        assert not arm_files.calls
    else:
        assert _load_fixture_arm(arm_files) is arm_files.scores


def test_load_arm_accepts_hash_verified_sidecar_beside_relocated_report(arm_files):
    _add_provenance(arm_files)
    arm_files.report['provenance']['path'] = '/former/run/tiny_arm_seed1.json.provenance'
    arm_files.report_path.write_text(json.dumps(arm_files.report))
    assert _load_fixture_arm(arm_files) is arm_files.scores


def test_load_arm_accepts_sidecar_from_actual_provenance_writer(arm_files):
    from run_provenance import begin_run

    _add_provenance(arm_files)
    output = arm_files.root / 'generated/tiny_arm_seed1.json'
    run_config = {**arm_files.config, 'seed': 1, 'train_steps_num': 1}
    captured = begin_run(
        config=run_config,
        params_path=arm_files.config_path,
        params_sha256=arm_files.report['params_sha256'],
        output_path=output,
        selection=['validation/ndcg@20'],
        experiment='tiny_confirm_seed1',
        source_root=Path(__file__).resolve().parents[1],
        evaluation_config=arm_files.config['callback']['callbacks'][0],
    )
    arm_files.report_path = output
    arm_files.report['provenance'] = {'path': captured['path'], 'sha256': captured['sha256']}
    output.write_text(json.dumps(arm_files.report))
    assert _load_fixture_arm(arm_files) is arm_files.scores
