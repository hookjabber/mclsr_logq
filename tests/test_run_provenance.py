"""Pre-training provenance and collision protection, using tiny local fixtures only."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

INPUT_NAMES = (
    'train_mclsr.txt',
    'train_sasrec.txt',
    'valid_history.txt',
    'valid_target.txt',
    'test_history.txt',
    'test_target.txt',
)


def _provenance():
    assert importlib.util.find_spec('run_provenance') is not None, 'pre-training provenance is not implemented'
    return importlib.import_module('run_provenance')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def run_inputs(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'src').mkdir(parents=True)
    (source / 'scripts').mkdir()
    (source / 'src/model.py').write_text('weight = 2\n')
    (source / 'src/irec').mkdir()
    (source / 'src/irec/__init__.py').write_text('')
    (source / 'scripts/untracked.py').write_text('seed = 7\n')
    (source / 'uv.lock').write_text('version = 1\n')
    (source / 'pyproject.toml').write_text('[project]\nname = "tiny"\n')
    data = tmp_path / 'data/Tiny'
    data.mkdir(parents=True)
    for index, name in enumerate(INPUT_NAMES):
        (data / name).write_text(f'{index + 1} 1 2 3\n')
    (data / 'preprocessing_manifest.json').write_text('{"protocol": "tiny"}\n')
    (data / 'item_counts.pkl').write_bytes(b'tiny count artifact')
    (data / 'user_counts.pkl').write_bytes(b'tiny user counts')
    workdir = tmp_path / 'run'
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    config = {
        'experiment_name': 'tiny',
        'seed': 7,
        'train_steps_num': 3,
        'dataset': {
            'type': 'graph',
            'dataset': {'type': 'mclsr', 'path_to_data_dir': '../data', 'name': 'Tiny'},
        },
        'loss': {
            'path_to_item_counts': '../data/Tiny/item_counts.pkl',
            'path_to_user_counts': '../data/Tiny/user_counts.pkl',
        },
    }
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(config))
    return SimpleNamespace(
        source=source,
        data=data,
        config=config,
        params=params,
        output=tmp_path / 'report.json',
        experiment='tiny_confirm_seed7',
        workdir=workdir,
    )


def _begin(inputs, **overrides):
    kwargs = dict(
        config=inputs.config,
        params_path=inputs.params,
        output_path=inputs.output,
        selection=['validation/ndcg@20'],
        experiment=inputs.experiment,
        source_root=inputs.source,
        imported_modules={'irec': _module_at(inputs.source / 'src/irec/__init__.py', package=True)},
    )
    kwargs.update(overrides)
    return _provenance().begin_run(**kwargs)


def test_captures_resolved_config_six_inputs_counts_and_environment(run_inputs):
    record = _begin(run_inputs)
    manifest = record['manifest']
    assert manifest['schema_version'] == 1
    assert manifest['phase'] == 'before_training'
    assert manifest['captured_at_utc'].endswith('+00:00')
    assert manifest['resolved_config'] == run_inputs.config
    assert manifest['selection'] == ['validation/ndcg@20']
    assert manifest['seed'] == 7
    assert manifest['step_limit'] == 3
    assert manifest['working_directory'] == str(run_inputs.workdir)
    assert manifest['params']['sha256'] == _sha(run_inputs.params)
    assert manifest['source']['imported_modules']['irec']['file'] == str(run_inputs.source / 'src/irec/__init__.py')
    dataset = manifest['datasets'][0]
    assert set(dataset['inputs']) == set(INPUT_NAMES)
    for name, metadata in dataset['inputs'].items():
        assert metadata['sha256'] == _sha(run_inputs.data / name)
        assert metadata['path'] == str(run_inputs.data / name)
    assert dataset['preprocessing_manifests']['preprocessing_manifest.json']['sha256'] == _sha(
        run_inputs.data / 'preprocessing_manifest.json'
    )
    for relative in ('../data/Tiny/item_counts.pkl', '../data/Tiny/user_counts.pkl'):
        assert manifest['artifacts'][relative]['sha256'] == _sha(relative)
    assert manifest['lockfiles']['uv.lock']['sha256'] == _sha(run_inputs.source / 'uv.lock')
    assert manifest['environment']['torch_version'] == torch.__version__
    assert manifest['environment']['python_version']
    assert 'cuda_available' in manifest['environment']
    assert 'gpus' in manifest['environment']
    assert record['sha256'] == _sha(record['path'])
    assert json.loads(Path(record['path']).read_text()) == manifest
    assert not run_inputs.output.exists()


def test_hashes_working_source_including_untracked_but_excludes_payloads_and_secrets(run_inputs, monkeypatch):
    for name in ('data', '.venv', 'results', 'checkpoints', 'src/data', 'scripts/__pycache__'):
        directory = run_inputs.source / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'payload.py').write_text('do not collect')
    (run_inputs.source / 'src/credentials.py').write_text('password = "private"')
    (run_inputs.source / '.env').write_text('TOKEN=private')
    (run_inputs.source / 'src/external.py').symlink_to(run_inputs.params)
    run_inputs.config['api_key'] = 'private-api-key'
    run_inputs.config['nested'] = {'access_token': 'private-token', 'secret': 'private-secret'}
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'private-environment-secret')
    record = _begin(run_inputs)
    sources = record['manifest']['source']['files']
    assert sources['src/model.py']['sha256'] == _sha(run_inputs.source / 'src/model.py')
    assert sources['scripts/untracked.py']['sha256'] == _sha(run_inputs.source / 'scripts/untracked.py')
    assert set(sources) == {'src/model.py', 'src/irec/__init__.py', 'scripts/untracked.py', 'pyproject.toml'}
    for secret in ('private-api-key', 'private-token', 'private-secret', 'private-environment-secret', 'password ='):
        assert secret not in Path(record['path']).read_text()
    assert run_inputs.config['api_key'] == 'private-api-key'


def test_non_git_source_is_explicit_not_falsely_clean(run_inputs):
    manifest = _begin(run_inputs)['manifest']
    assert manifest['source']['git_commit'] is None
    assert manifest['source']['git_dirty'] is None
    assert manifest['source']['git_root'] is None


def test_git_metadata_uses_source_root_not_temporary_working_directory(run_inputs):
    manifest = _begin(run_inputs, source_root=ROOT, imported_modules={'irec': importlib.import_module('irec')})[
        'manifest'
    ]
    expected = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert manifest['source']['git_commit'] == expected
    assert manifest['source']['git_root'] == str(ROOT)
    assert isinstance(manifest['source']['git_dirty'], bool)
    assert manifest['source']['files']['scripts/train_confirmatory.py']['sha256'] == _sha(
        ROOT / 'scripts/train_confirmatory.py'
    )


@pytest.mark.parametrize('existing', ['report', 'manifest', 'checkpoint', 'namespace'])
def test_rejects_existing_outputs_without_modifying_them(run_inputs, existing):
    manifest_path = Path(str(run_inputs.output) + '.provenance')
    checkpoint_dir = run_inputs.workdir / 'checkpoints'
    checkpoint_dir.mkdir()
    paths = {
        'report': run_inputs.output,
        'manifest': manifest_path,
        'checkpoint': checkpoint_dir / f'{run_inputs.experiment}_best_validation_other.pth',
        'namespace': checkpoint_dir / f'{run_inputs.experiment}.provenance.json',
    }
    paths[existing].write_text('original artifact')
    before = {path: path.read_bytes() for path in run_inputs.workdir.parent.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        _begin(run_inputs)
    after = {path: path.read_bytes() for path in run_inputs.workdir.parent.rglob('*') if path.is_file()}
    assert after == before


def test_reservation_rejects_same_namespace_with_new_output(run_inputs):
    _begin(run_inputs)
    with pytest.raises(FileExistsError):
        _begin(run_inputs, output_path=run_inputs.output.with_name('different.json'))


def test_same_seed_and_selection_can_repeat_in_different_cwd(run_inputs, monkeypatch):
    first = _begin(run_inputs)
    second_cwd = run_inputs.workdir.with_name('run2')
    second_cwd.mkdir()
    monkeypatch.chdir(second_cwd)
    second = _begin(run_inputs, output_path=run_inputs.output.with_name('report2.json'))
    assert first['path'] != second['path']
    assert second['manifest']['datasets'] == first['manifest']['datasets']
    assert second['manifest']['seed'] == first['manifest']['seed']


def test_missing_input_fails_before_reserving_any_output(run_inputs):
    (run_inputs.data / 'test_target.txt').unlink()
    with pytest.raises(FileNotFoundError, match='test_target.txt'):
        _begin(run_inputs)
    assert not Path(str(run_inputs.output) + '.provenance').exists()
    assert not (run_inputs.workdir / 'checkpoints').exists()


def test_absent_optional_preprocessing_manifest_is_explicit(run_inputs):
    (run_inputs.data / 'preprocessing_manifest.json').unlink()
    assert _begin(run_inputs)['manifest']['datasets'][0]['preprocessing_manifests'] == {}


def test_captures_prepare_dataset_preprocessing_json(run_inputs):
    manifest_path = run_inputs.data / 'preprocessing.json'
    (run_inputs.data / 'preprocessing_manifest.json').rename(manifest_path)
    dataset = _begin(run_inputs)['manifest']['datasets'][0]
    assert dataset['preprocessing_manifests']['preprocessing.json']['sha256'] == _sha(manifest_path)


def test_source_collection_is_bounded_and_fails_instead_of_silently_omitting(run_inputs, monkeypatch):
    monkeypatch.setattr(_provenance(), 'MAX_SOURCE_BYTES', 1)
    with pytest.raises(ValueError, match='source.*limit'):
        _begin(run_inputs)
    assert not Path(str(run_inputs.output) + '.provenance').exists()


def test_captures_artifacts_used_only_by_post_training_evaluation(run_inputs):
    evaluation_counts = run_inputs.data / 'evaluation_counts.pkl'
    evaluation_counts.write_bytes(b'evaluation-only counts')
    manifest = _begin(run_inputs, evaluation_config={'path_to_counts': str(evaluation_counts)})['manifest']
    assert manifest['artifacts'][str(evaluation_counts)]['sha256'] == _sha(evaluation_counts)


def test_report_write_never_overwrites_an_existing_file(run_inputs):
    run_inputs.output.write_text('original report')
    with pytest.raises(FileExistsError):
        _provenance().write_report(run_inputs.output, {'new': 'report'})
    assert run_inputs.output.read_text() == 'original report'


def test_sidecar_does_not_match_legacy_report_globs(run_inputs):
    output = run_inputs.output.with_name('toys_baseline_seed7.json')
    record = _begin(run_inputs, output_path=output)
    _provenance().write_report(output, {'results': {}})
    assert Path(record['path']).parent == output.parent
    assert sorted(output.parent.glob('toys_*_seed*.json')) == [output]
    assert sorted(output.parent.glob('*.json')) == sorted([output, run_inputs.params])


def _validate_origins(source_root, modules):
    validator = getattr(_provenance(), 'validate_imported_sources', None)
    assert callable(validator), 'imported-source validation is not implemented'
    return validator(source_root, imported_modules=modules)


def _module_at(path, package=False):
    module = SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(origin=str(path)))
    if package:
        module.__path__ = [str(path.parent)]
    return module


@pytest.fixture
def imported_modules(run_inputs):
    package_dir = run_inputs.source / 'src/irec'
    package_dir.mkdir(exist_ok=True)
    (package_dir / '__init__.py').write_text('')
    (package_dir / 'model.py').write_text('weight = 2\n')
    return {
        'irec': _module_at(package_dir / '__init__.py', package=True),
        'irec.model': _module_at(package_dir / 'model.py'),
    }


def test_imported_sources_accept_matching_modules_without_changing_imports(run_inputs, imported_modules):
    paths_before = list(sys.path)
    records = _validate_origins(run_inputs.source, imported_modules)
    assert set(records) == set(imported_modules)
    for name, module in imported_modules.items():
        assert records[name]['file'] == str(Path(module.__file__).resolve())
        assert records[name]['spec_origin'] == str(Path(module.__spec__.origin).resolve())
    assert sys.path == paths_before


@pytest.mark.parametrize(
    'module_name,attribute', [('irec', '__file__'), ('irec.model', '__file__'), ('irec.model', 'origin')]
)
def test_imported_sources_reject_other_checkout_origins(run_inputs, imported_modules, module_name, attribute):
    original = imported_modules[module_name]
    foreign = run_inputs.source.parent / 'old-checkout/src/irec/model.py'
    foreign.parent.mkdir(parents=True)
    foreign.write_text('old_weight = 1\n')
    target = original.__spec__ if attribute == 'origin' else original
    setattr(target, attribute, str(foreign))
    with pytest.raises(RuntimeError, match='Imported-source mismatch') as error:
        _validate_origins(run_inputs.source, imported_modules)
    assert module_name in str(error.value)
    assert str(foreign) in str(error.value)
    assert str(run_inputs.source / 'src/irec') in str(error.value)
    assert getattr(target, attribute) == str(foreign)


def test_imported_sources_reject_external_package_search_path(run_inputs, imported_modules):
    imported_modules['irec'].__path__.append(str(run_inputs.source.parent / 'old-checkout/src/irec'))
    with pytest.raises(RuntimeError, match='Imported-source mismatch'):
        _validate_origins(run_inputs.source, imported_modules)


@pytest.mark.parametrize('defect', ['missing_package', 'missing_file', 'missing_spec_origin'])
def test_imported_sources_fail_closed_when_origins_are_unknown(run_inputs, imported_modules, defect):
    if defect == 'missing_package':
        del imported_modules['irec']
    elif defect == 'missing_file':
        imported_modules['irec.model'].__file__ = None
    else:
        imported_modules['irec.model'].__spec__.origin = None
    with pytest.raises(RuntimeError, match='Cannot verify imported source'):
        _validate_origins(run_inputs.source, imported_modules)


def test_begin_run_detects_actual_import_mismatch_without_pythonpath(run_inputs, monkeypatch):
    importlib.import_module('irec')
    monkeypatch.delenv('PYTHONPATH', raising=False)
    with pytest.raises(RuntimeError, match='Imported-source mismatch'):
        _provenance().begin_run(
            config=run_inputs.config,
            params_path=run_inputs.params,
            output_path=run_inputs.output,
            selection=['validation/ndcg@20'],
            experiment=run_inputs.experiment,
            source_root=run_inputs.source,
        )
    assert not (run_inputs.workdir / 'checkpoints').exists()
    assert not Path(str(run_inputs.output) + '.provenance').exists()


def test_copied_provenance_rejects_old_editable_import_without_pythonpath(run_inputs):
    copied_scripts = run_inputs.source / 'scripts'
    (copied_scripts / 'run_provenance.py').write_bytes((ROOT / 'scripts/run_provenance.py').read_bytes())
    package_dir = run_inputs.source / 'src/irec'
    package_dir.mkdir(exist_ok=True)
    (package_dir / '__init__.py').write_text('')
    script = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
import irec
sys.path.pop(0)
sys.path.insert(0, {str(copied_scripts)!r})
import run_provenance
run_provenance.begin_run(
    config={run_inputs.config!r}, params_path={str(run_inputs.params)!r},
    output_path={str(run_inputs.output)!r}, selection=['validation/ndcg@20'],
    experiment='copied_source_check',
)
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-B', '-c', script],
        cwd=run_inputs.workdir,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert 'Imported-source mismatch' in result.stderr, result.stderr
    assert str(ROOT / 'src/irec') in result.stderr
    assert not (run_inputs.workdir / 'checkpoints').exists()


@pytest.mark.parametrize('late_foreign_import', [False, True])
def test_confirmatory_manifest_exists_before_train_and_report_uses_start_hashes(
    run_inputs, monkeypatch, late_foreign_import
):
    runner = importlib.import_module('train_confirmatory')
    config = copy.deepcopy(run_inputs.config)
    config.update(
        dataloader={'train': {}, 'validation': {}},
        model={},
        optimizer={},
        callback={
            'type': 'composite',
            'callbacks': [
                {'type': 'validation', 'metrics': {'ndcg@20': {}}},
                {'type': 'eval', 'metrics': {}, 'pred_prefix': 'predictions', 'labels_prefix': 'labels'},
            ],
        },
    )
    run_inputs.params.write_text(json.dumps(config))
    params_hash = _sha(run_inputs.params)
    counts_hash = _sha(run_inputs.data / 'item_counts.pkl')
    log_dir = run_inputs.workdir / 'logs'
    log_dir.mkdir()
    monkeypatch.setattr(
        runner.irec.utils.tensorboards, 'TensorboardWriter', lambda *args: SimpleNamespace(log_dir=str(log_dir))
    )
    monkeypatch.setattr(runner.irec.utils.tensorboards, 'GLOBAL_TENSORBOARD_WRITER', None)

    def fake_dataset(*args):
        if late_foreign_import:
            monkeypatch.setitem(sys.modules, 'irec.foreign', _module_at(run_inputs.source / 'src/model.py'))
        return SimpleNamespace(meta={}, get_samplers=lambda: (1, 2, 3))

    monkeypatch.setattr(runner.BaseDataset, 'create_from_config', fake_dataset)
    monkeypatch.setattr(runner.BaseDataloader, 'create_from_config', lambda *args, **kwargs: [])
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(runner.BaseModel, 'create_from_config', lambda *args, **kwargs: model)
    monkeypatch.setattr(runner.BaseLoss, 'create_from_config', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.BaseOptimizer, 'create_from_config', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.BaseCallback, 'create_from_config', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'evaluate', lambda **kwargs: {'ndcg@20': 0.5})
    seen = {}

    def fake_train(**kwargs):
        assert not late_foreign_import, 'training must not start with a foreign module imported during setup'
        manifest_path = Path(str(run_inputs.output) + '.provenance')
        assert manifest_path.exists(), 'manifest must exist before train() is called'
        seen.update(json.loads(manifest_path.read_text()))
        assert seen['params']['sha256'] == params_hash
        assert seen['resolved_config']['seed'] == 11
        assert seen['resolved_config']['train_steps_num'] == kwargs['step_cnt'] == 2
        assert seen['selection'] == kwargs['best_metric'] == ['validation/ndcg@20']
        assert all(entry['type'] != 'eval' for entry in seen['resolved_config']['callback']['callbacks'])
        assert seen['evaluation_config']['type'] == 'eval'
        assert seen['source']['imported_modules']['irec']['file'] == str(Path(runner.irec.__file__).resolve())
        run_inputs.params.write_text('{}')
        (run_inputs.data / 'item_counts.pkl').write_bytes(b'changed after training started')
        return {'validation/ndcg@20': {'state': model.state_dict(), 'value': 0.5, 'step': 2, 'epoch': 1}}

    monkeypatch.setattr(runner, 'train', fake_train)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_confirmatory.py',
            '--params',
            str(run_inputs.params),
            '--seed',
            '11',
            '--max-steps',
            '2',
            '--select',
            'validation/ndcg@20',
            '--output',
            str(run_inputs.output),
        ],
    )
    if late_foreign_import:
        with pytest.raises(RuntimeError, match='Imported-source mismatch'):
            runner.main()
        assert not run_inputs.output.exists()
        return
    runner.main()
    report = json.loads(run_inputs.output.read_text())
    assert report['params_sha256'] == params_hash != _sha(run_inputs.params)
    assert report['artifact_sha256']['../data/Tiny/item_counts.pkl'] == counts_hash
    assert report['provenance']['sha256'] == _sha(report['provenance']['path'])
    assert report['git_commit'] == seen['source']['git_commit']
    assert report['git_dirty'] == seen['source']['git_dirty']
    assert report['torch_version'] == seen['environment']['torch_version']
    assert report['results']['validation/ndcg@20']['checkpoint_sha256'] == _sha(
        report['results']['validation/ndcg@20']['checkpoint']
    )
    assert runner.get_inference_config(json.loads((log_dir / 'config.json').read_text()), 'eval')['type'] == 'eval'
