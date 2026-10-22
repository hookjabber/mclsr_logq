import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def followup_module():
    assert importlib.util.find_spec('prepare_logq_followup') is not None, 'follow-up preparer is missing'
    return importlib.import_module('prepare_logq_followup')


def parent_fixture(tmp_path):
    import run_component_screen as screen

    root = tmp_path / 'parent'
    (root / 'configs').mkdir(parents=True)
    base = json.loads((ROOT / 'configs/train/clothing64/05_full_paper.json').read_text())
    configs = screen.build_configs(base, 'parent', '/frozen/users.pkl', 65536, 256)
    arms = ('masked_dq1_auxq0', 'masked_dq1_user_feature_q1')
    records = {}
    for arm in arms:
        path = root / 'configs' / f'{arm}.json'
        screen.write_json(path, configs[arm])
        records[arm] = {'path': str(path), 'sha256': screen.sha256(path)}
        result = root / 'runs' / arm / 'result.json'
        result.parent.mkdir(parents=True)
        screen.write_json(result, {'status': 'training_finished'})
    manifest = {
        'root': str(root),
        'repo_root': str(tmp_path / 'source'),
        'seed': 1,
        'steps': 65536,
        'validation_every': 256,
        'selection': screen.SELECTION,
        'test_evaluation': False,
        'arms': list(arms),
        'configs': records,
        'source_sha256': {'src/model.py': 'frozen'},
        'tool_sha256': {},
        'artifacts_sha256': {},
        'launcher': '/frozen/native.sh',
    }
    path = root / 'screen.json'
    screen.write_json(path, manifest)
    return screen, path, manifest


def fake_validation(monkeypatch, screen):
    monkeypatch.setattr(screen, 'validate_screen', lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(screen, 'validate_result', lambda *args: None)


def test_preparation_preserves_parent_and_writes_exactly_four_unstarted_runs(tmp_path, monkeypatch):
    module = followup_module()
    screen, parent_path, parent = parent_fixture(tmp_path)
    fake_validation(monkeypatch, screen)
    digest = screen.sha256(parent_path)
    output = tmp_path / 'logq_followup'
    index_path = module.prepare_followup(parent_path, digest, output, screen)
    index = json.loads(index_path.read_text())
    assert index['status'] == 'prepared_not_started'
    assert index['seeds'] == [2, 3]
    assert index['parent']['sha256'] == digest
    assert len(index['queues']) == 2
    names = set()
    for record in index['queues']:
        manifest = json.loads(Path(record['manifest']).read_text())
        assert record['sha256'] == screen.sha256(record['manifest'])
        assert manifest['seed'] in (2, 3)
        for key in (
            'repo_root',
            'steps',
            'validation_every',
            'selection',
            'test_evaluation',
            'source_sha256',
            'launcher',
        ):
            assert manifest[key] == parent[key]
        assert manifest['arms'] == parent['arms']
        for arm in manifest['arms']:
            old = json.loads(Path(parent['configs'][arm]['path']).read_text())
            new = json.loads(Path(manifest['configs'][arm]['path']).read_text())
            assert old['experiment_name'] != new['experiment_name']
            names.add(new['experiment_name'])
            new['experiment_name'] = old['experiment_name']
            assert old == new
            assert manifest['configs'][arm]['sha256'] == screen.sha256(manifest['configs'][arm]['path'])
        assert not (Path(manifest['root']) / 'queue.reservation.json').exists()
        assert not (Path(manifest['root']) / 'runs').exists()
    assert len(names) == 4
    assert screen.sha256(parent_path) == digest
    before = {str(path): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        module.prepare_followup(parent_path, digest, output, screen)
    assert before == {str(path): path.read_bytes() for path in output.rglob('*') if path.is_file()}


@pytest.mark.parametrize('change', ['mask', 'model', 'lambda', 'order'])
def test_rejects_unmatched_parent_pair_before_writing(tmp_path, monkeypatch, change):
    module = followup_module()
    screen, path, parent = parent_fixture(tmp_path)
    fake_validation(monkeypatch, screen)
    params = Path(parent['configs']['masked_dq1_user_feature_q1']['path'])
    config = json.loads(params.read_text())
    if change == 'mask':
        config['loss']['losses'][1]['mask_false_negatives'] = False
    elif change == 'model':
        config['model']['unexpected'] = True
    elif change == 'lambda':
        config['loss']['losses'][2]['logq_lambda'] = 0.5
    else:
        config['loss']['losses'].reverse()
    params.write_text(json.dumps(config))
    parent['configs']['masked_dq1_user_feature_q1']['sha256'] = screen.sha256(params)
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match='matched'):
        module.prepare_followup(path, screen.sha256(path), tmp_path / 'followup', screen)
    assert not (tmp_path / 'followup').exists()


@pytest.mark.parametrize('failure', ['hash', 'result', 'test', 'seed', 'namespace'])
def test_preparation_fails_closed_before_outputs(tmp_path, monkeypatch, failure):
    module = followup_module()
    screen, path, parent = parent_fixture(tmp_path)
    fake_validation(monkeypatch, screen)
    output = tmp_path / 'followup'
    if failure == 'test':
        parent['test_evaluation'] = True
    elif failure == 'seed':
        parent['seed'] = 2
    path.write_text(json.dumps(parent))
    digest = screen.sha256(path)
    if failure == 'hash':
        digest = '0' * 64
    elif failure == 'result':

        def reject(*args):
            raise ValueError('parent result is incomplete')

        monkeypatch.setattr(screen, 'validate_result', reject)
    elif failure == 'namespace':
        namespace = Path(parent['repo_root']) / 'checkpoints' / 'followup_seed2_masked_dq1_auxq0_existing.pth'
        namespace.parent.mkdir(parents=True)
        namespace.touch()
    with pytest.raises((ValueError, FileExistsError)):
        module.prepare_followup(path, digest, output, screen)
    assert not output.exists()
