import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_dataset


def _module(name):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f'{name} must implement the frozen protocol study'
    return __import__(name)


@pytest.fixture
def inputs(tmp_path):
    sequences = [(user, [(user + offset) % 35 + 1 for offset in range(25)]) for user in range(1, 31)]
    raw = tmp_path / 'source.txt'
    raw.write_text(''.join(' '.join(map(str, [user, *items])) + '\n' for user, items in sequences))
    reference = tmp_path / 'reference'
    prepare_dataset.write_files(sequences, *prepare_dataset.split_users(sequences), reference)
    return sequences, raw, reference, tmp_path / 'study'


def _prepare(inputs, **overrides):
    sequences, raw, reference, output = inputs
    options = {
        'sequences': sequences,
        'raw_path': raw,
        'legacy_reference': reference,
        'output': output,
        'repo_root': ROOT,
        'template_path': ROOT / 'configs/train/clothing64/05_full_paper.json',
        'steps': 64,
        'validation_every': 32,
    }
    options.update(overrides)
    return _module('prepare_protocol_study').prepare_study(**options)


def test_study_separates_data_and_proposal_with_fixed_budget(inputs):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    assert study['steps'] == 64
    assert study['selection'] == ['validation/ndcg@20']
    assert study['evaluation_equivalent'] is True
    assert set(study['configs']) == {'legacy_l00', 'legacy_proxy_l10', 'legacy_target_l10', 'v2_l00', 'v2_target_l10'}
    assert len(study['runs']) == 15
    assert len({(run['arm'], run['seed']) for run in study['runs']}) == 15
    configs = {name: json.loads(Path(record['path']).read_text()) for name, record in study['configs'].items()}
    reference = configs['legacy_l00']
    for config in configs.values():
        assert config['train_steps_num'] == config['epochs_threshold'] == 64
        assert 'train_epochs_num' not in config
        assert config['model'] == reference['model']
        assert config['optimizer'] == reference['optimizer']
        assert config['dataloader'] == reference['dataloader']
        assert config['loss']['losses'][1:] == reference['loss']['losses'][1:]
        for callback in config['callback']['callbacks']:
            if callback['type'] in ('validation', 'eval'):
                assert callback['on_step'] == 32
                assert set(callback['metrics']) == {'ndcg@20', 'recall@1000'}
    for arm in ('legacy_target_l10', 'v2_target_l10'):
        loss = configs[arm]['loss']['losses'][0]
        assert loss['logq_lambda'] == 1.0
        assert loss['expected_counts_role'] == 'target'
        assert loss['expected_counts_max_len'] == (21 if arm.startswith('v2') else 20)
    assert 'expected_counts_role' not in configs['legacy_proxy_l10']['loss']['losses'][0]
    assert study['data']['v2']['train_targets'] == 24 * 24
    assert study['data']['legacy']['train_targets'] == 24 * 19


def test_study_does_not_overwrite_existing_outputs(inputs):
    path = _prepare(inputs)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _prepare(inputs)
    assert path.read_bytes() == before


def test_changed_legacy_reference_blocks_study(inputs):
    inputs[2].joinpath('test_target.txt').write_text('999 1\n')
    with pytest.raises(ValueError, match='legacy'):
        _prepare(inputs)
    assert not inputs[3].joinpath('study.json').exists()


def test_logical_evaluation_comparison_respects_inverse_mapping(inputs):
    sequences, raw, _, output = inputs
    sequences = [(user * 3, [item * 2 for item in items]) for user, items in sequences]
    reference = raw.parent / 'remapped'
    prepare_dataset.write_files(sequences, *prepare_dataset.split_users(sequences), reference)
    versioned = output / 'data/v2'
    prepare_dataset.write_files(sequences, *prepare_dataset.split_users(sequences), versioned, protocol='windowed-v2')
    verifier = _module('prepare_protocol_study').verify_evaluation
    assert verifier(reference, versioned) is True
    v2 = versioned / 'test_target.txt'
    v2.write_text(v2.read_text().replace(' 1', ' 2', 1) + '999 1\n')
    with pytest.raises(ValueError, match='evaluation'):
        verifier(reference, versioned)


def test_study_rejects_different_candidate_catalogues(inputs):
    sequences, raw, _, _ = inputs
    sequences = [(user, [item * 2 for item in items]) for user, items in sequences]
    reference = raw.parent / 'sparse'
    prepare_dataset.write_files(sequences, *prepare_dataset.split_users(sequences), reference)
    with pytest.raises(ValueError, match='candidate catalogues differ'):
        _prepare(inputs, sequences=sequences, legacy_reference=reference)


def test_queue_partitions_all_runs_once(inputs):
    path = _prepare(inputs)
    runner = _module('run_protocol_study')
    study = runner.validate_study(path)
    first = runner.partition_runs(study, 0, 2)
    second = runner.partition_runs(study, 1, 2)
    assert len(first) == 8 and len(second) == 7
    assert {(run['arm'], run['seed']) for run in first}.isdisjoint({(run['arm'], run['seed']) for run in second})
    assert sorted(first + second, key=lambda run: run['index']) == study['runs']
    with pytest.raises(ValueError):
        runner.partition_runs(study, 2, 2)
    with pytest.raises(ValueError, match='workers'):
        runner.partition_runs(study, 1, 4)


@pytest.mark.parametrize('changed', ['config', 'dataset', 'source'])
def test_queue_refuses_changed_frozen_inputs(inputs, changed, monkeypatch):
    path = _prepare(inputs)
    runner = _module('run_protocol_study')
    study = json.loads(path.read_text())
    if changed == 'source':
        altered = copy.deepcopy(study['source_sha256'])
        altered['scripts/train_confirmatory.py'] = '0' * 64
        monkeypatch.setattr(runner, 'source_hashes', lambda root: altered)
    else:
        target = (
            Path(study['configs']['legacy_l00']['path'])
            if changed == 'config'
            else inputs[3] / 'data/legacy/test_target.txt'
        )
        target.write_text(target.read_text() + '\n')
    with pytest.raises(ValueError, match='changed|mismatch'):
        runner.validate_study(path)


def test_queue_refuses_new_preprocessing_metadata(inputs):
    path = _prepare(inputs)
    (inputs[3] / 'data/legacy/preprocessing.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata'):
        _module('run_protocol_study').validate_study(path)


def _completed_report(study, run, output):
    preparer = _module('prepare_protocol_study')
    config_record = study['configs'][run['arm']]
    config = json.loads(Path(config_record['path']).read_text())
    checkpoint = output.with_suffix('.pth')
    checkpoint.write_bytes(b'synthetic checkpoint')
    provenance = output.with_suffix('.provenance')
    source = {name: {'sha256': digest} for name, digest in study['source_sha256'].items()}
    preparer.write_json(
        provenance,
        {
            'seed': run['seed'],
            'selection': study['selection'],
            'step_limit': study['steps'],
            'params': {'sha256': config_record['sha256']},
            'source': {'files': source},
        },
    )
    report = {
        'seed': run['seed'],
        'step_limit': study['steps'],
        'optimizer_updates': study['steps'],
        'params_sha256': config_record['sha256'],
        'experiment': f'{config["experiment_name"]}_confirm_seed{run["seed"]}',
        'provenance': {'path': str(provenance), 'sha256': preparer.sha256(provenance)},
        'results': {
            'validation/ndcg@20': {
                'selection_metric': 'validation/ndcg@20',
                'checkpoint': str(checkpoint),
                'checkpoint_sha256': preparer.sha256(checkpoint),
                'step': 0,
                'validation_value': 0.1,
                'test': {'ndcg@20': 0.1, 'recall@1000': 0.2},
            }
        },
    }
    preparer.write_json(output, report)
    return report


@pytest.mark.parametrize('changed', ['seed', 'optimizer_updates', 'params_sha256', 'checkpoint', 'provenance'])
def test_completed_report_must_match_frozen_run(inputs, changed):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    run = study['runs'][0]
    output = path.with_name('report.json')
    report = _completed_report(study, run, output)
    runner = _module('run_protocol_study')
    assert hasattr(runner, 'validate_report'), 'completed reports need semantic validation'
    runner.validate_report(study, run, output)
    if changed in ('checkpoint', 'provenance'):
        target = (
            report['results']['validation/ndcg@20']['checkpoint']
            if changed == 'checkpoint'
            else report['provenance']['path']
        )
        Path(target).write_bytes(b'changed')
    else:
        report[changed] = -1
        output.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runner.validate_report(study, run, output)


@pytest.mark.parametrize('returncode', [0, 7])
def test_queue_executes_selected_runs_and_stops_on_failure(inputs, monkeypatch, returncode):
    path = _prepare(inputs)
    study = json.loads(path.read_text())
    runner = _module('run_protocol_study')
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'] == str(ROOT)
        assert kwargs['env']['PYTHONPATH'] == str(ROOT / 'src')
        assert command[command.index('--max-steps') + 1] == '64'
        assert command[command.index('--select') + 1] == 'validation/ndcg@20'
        if returncode == 0:
            config_path = command[command.index('--params') + 1]
            run = {'arm': Path(config_path).stem, 'seed': int(command[command.index('--seed') + 1])}
            _completed_report(study, run, Path(command[command.index('--output') + 1]))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr(sys, 'argv', ['run_protocol_study.py', '--study', str(path), '--worker', '0'])
    if returncode:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == returncode
    else:
        runner.main()
    assert len(calls) == (1 if returncode else 8)
    events = [json.loads(line) for line in path.with_name('queue_0.events.jsonl').read_text().splitlines()]
    assert len(events) == len(calls) * 2
    assert events[-1]['status'] == ('failed' if returncode else 'complete')
    with pytest.raises(FileExistsError):
        runner.main()
