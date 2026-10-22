import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def screen_module():
    assert importlib.util.find_spec('run_component_screen') is not None, 'component screen runner is missing'
    return importlib.import_module('run_component_screen')


def baseline():
    return json.loads((ROOT / 'configs/train/clothing64/05_full_paper.json').read_text())


def test_screen_has_matched_objective_and_correction_controls():
    module = screen_module()
    base = baseline()
    original = copy.deepcopy(base)
    configs = module.build_configs(base, 'screen', '/tmp/user.pkl', 65536, 256)
    assert base == original
    assert len(configs) == 15
    for config in configs.values():
        assert config['model'] == base['model']
        assert config['dataset'] == base['dataset']
        assert config['optimizer'] == base['optimizer']
        assert config['dataloader'] == base['dataloader']
        assert config['train_steps_num'] == config['epochs_threshold'] == 65536
        assert 'train_epochs_num' not in config
        prefixes = {loss['output_prefix'] for loss in config['loss']['losses']} | {'loss'}
        for callback in config['callback']['callbacks']:
            if callback['type'] == 'metric':
                assert callback['loss_prefix'] in prefixes
            elif callback['type'] in ('validation', 'eval'):
                assert callback['on_step'] == 256
    assert len(configs['downstream_only_dq1']['loss']['losses']) == 1
    for component in ('interest', 'user_feature', 'item_feature'):
        arm = configs[f'no_{component}_dq1']
        assert len(arm['loss']['losses']) == 3
        assert f'contrastive_{component}_loss' not in {loss['output_prefix'] for loss in arm['loss']['losses']}
    control = configs['masked_dq1_auxq0']['loss']['losses']
    assert all(loss['mask_false_negatives'] for loss in control[1:])
    for component in ('interest', 'user_feature', 'item_feature'):
        losses = configs[f'masked_dq1_{component}_q1']['loss']['losses']
        for reference, loss in zip(control, losses):
            expected = copy.deepcopy(reference)
            if loss['output_prefix'] == f'contrastive_{component}_loss':
                expected['logq_lambda'] = 1.0
            assert loss == expected
    assert all(loss['logq_lambda'] == 1 for loss in configs['masked_dq1_all_q1']['loss']['losses'])


def test_invalid_screen_budget_is_rejected():
    module = screen_module()
    for steps, interval in ((0, 1), (10, 0), (2, 3), (10, 2.5)):
        with pytest.raises(ValueError):
            module.build_configs(baseline(), 'screen', '/tmp/user.pkl', steps, interval)


def test_optional_source_snapshot_allows_only_the_two_audited_loss_repairs(tmp_path):
    module = screen_module()
    assert hasattr(module, 'validate_source_overlay'), 'audited source overlay guard is missing'
    original = tmp_path / 'original'
    candidate = tmp_path / 'candidate'
    for root in (original, candidate):
        (root / 'src/irec/loss').mkdir(parents=True)
        (root / 'src/irec/loss/counts.py').write_text('original\n')
        (root / 'src/other.py').write_text('unchanged\n')
    from prepare_protocol_study import source_hashes

    study = {'repo_root': str(original), 'source_sha256': source_hashes(original)}
    (candidate / 'src/irec/loss/counts.py').write_text('repaired\n')
    root, hashes, changed = module.validate_source_overlay(study, candidate)
    assert root == candidate.resolve()
    assert hashes == source_hashes(candidate)
    assert changed == ['src/irec/loss/counts.py']
    (candidate / 'src/other.py').write_text('unexpected\n')
    with pytest.raises(ValueError, match='audited'):
        module.validate_source_overlay(study, candidate)


def test_user_proposals_use_v2_ladder_ids_and_do_not_smooth_missing_users(tmp_path):
    module = screen_module()
    source = tmp_path / 'train_mclsr.txt'
    source.write_text('2 1 3\n2 1 3 4\n4 2 3\n')
    output = tmp_path / 'users.pkl'
    module.write_user_counts(source, output, num_users=5, max_len=21)
    with output.open('rb') as stream:
        record = pickle.load(stream)
    assert list(record['counts']) == [0, 0, 2, 0, 1, 0, 0]
    assert record['denominator'] == record['num_lines'] == 3
    assert record['role'] == 'user-row' and record['max_len'] == 21
    with pytest.raises(FileExistsError):
        module.write_user_counts(source, output, num_users=5, max_len=21)


def test_validation_capture_never_builds_test_loader():
    from test_diagnose_training import fake_runner

    runner, calls = fake_runner()
    original = runner.train
    report = screen_module().collect_validation(runner, [])
    assert calls == [0, 1, 2]
    assert runner.train is original
    assert report['test_evaluated'] is False
    assert report['optimizer_updates'] == 3
    assert report['best_validation']['validation/ndcg@20'] == {'step': 1, 'value': 0.2}
    assert 'state' not in json.dumps(report)


def test_validation_capture_rejects_nonfinite_metrics():
    from test_diagnose_training import fake_runner

    runner, calls = fake_runner()
    original = runner.train

    def broken(**kwargs):
        result = original(**kwargs)
        result['validation/ndcg@20']['value'] = float('nan')
        return result

    runner.train = broken
    with pytest.raises(ValueError, match='finite'):
        screen_module().collect_validation(runner, [])
    assert 'test_loader' not in calls


@pytest.mark.parametrize('codes,expected_attempts', [([139, 0, 0], 3), ([139, 139, 0], 2), ([1, 0, 0], 1)])
def test_queue_records_failure_without_retry_and_advances_only_native_failures(
    tmp_path, monkeypatch, codes, expected_attempts
):
    module = screen_module()
    manifest = {'root': str(tmp_path), 'arms': ['first', 'second', 'third'], 'repo_root': str(ROOT), 'seed': 1}
    manifest_path = tmp_path / 'screen.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, 'validate_screen', lambda path: manifest)
    monkeypatch.setattr(module, 'validate_result', lambda *args: None)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=codes[len(commands) - 1])

    monkeypatch.setattr(module.subprocess, 'run', execute)
    assert module.run_queue(manifest_path) == 1
    assert len(commands) == expected_attempts
    events = [json.loads(line) for line in (tmp_path / 'queue.events.jsonl').read_text().splitlines()]
    assert [event['arm'] for event in events if event['status'] == 'running'] == manifest['arms'][:expected_attempts]
    assert events[1]['status'] == 'failed'
    with pytest.raises(FileExistsError):
        module.run_queue(manifest_path)


def test_real_screen_preserves_source_and_never_evaluates_test(tmp_path, monkeypatch):
    import prepare_dataset
    from prepare_protocol_study import prepare_study, source_hashes

    module = screen_module()
    monkeypatch.setattr(sys, 'path', list(sys.path))
    frozen = tmp_path / 'source'
    for directory in ('src', 'scripts'):
        shutil.copytree(ROOT / directory, frozen / directory, ignore=shutil.ignore_patterns('__pycache__'))
    sequences = [(user, [(user + offset) % 1001 + 1 for offset in range(25)]) for user in range(1, 1002)]
    raw = tmp_path / 'raw.txt'
    raw.write_text(''.join(' '.join(map(str, [user, *items])) + '\n' for user, items in sequences))
    reference = tmp_path / 'reference'
    prepare_dataset.write_files(sequences, *prepare_dataset.split_users(sequences), reference)
    template = baseline()
    template_path = tmp_path / 'template.json'
    template_path.write_text(json.dumps(template))
    study = prepare_study(
        sequences, raw, reference, tmp_path / 'study', frozen, template_path, steps=4, validation_every=2
    )
    original_hashes = source_hashes(frozen)
    repaired = tmp_path / 'repaired_source'
    shutil.copytree(frozen, repaired)
    with (repaired / 'src/irec/loss/counts.py').open('a') as destination:
        destination.write('\n')
    path = module.prepare_screen(study, tmp_path / 'screen', steps=4, source_root=repaired)
    manifest = module.validate_screen(path)
    arm = 'masked_dq1_all_q1'
    outcome = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/run_component_screen.py'), 'arm', '--manifest', str(path), '--arm', arm],
        cwd=tmp_path,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='',
            PYTHONPATH=os.pathsep.join([str(frozen / 'src'), str(frozen / 'scripts')]),
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    report = module.validate_result(manifest, arm, path)
    assert report['last_validation']['step'] == 2
    assert report['optimizer_updates'] == 4
    assert report['test_evaluated'] is False
    assert not (path.parent / 'runs' / arm / 'unused_test_report.json').exists()
    assert source_hashes(frozen) == original_hashes
    with pytest.raises(FileExistsError):
        module.prepare_screen(study, path.parent, steps=4)
    Path(manifest['configs'][arm]['path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        module.validate_screen(path)
