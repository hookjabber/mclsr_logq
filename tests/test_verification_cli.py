import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args, cwd, inject_mclsr_mismatch=False):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='')
    for variable in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'POLARS_MAX_THREADS',
    ):
        env[variable] = '1'
    command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, args)]
    if inject_mclsr_mismatch:
        bootstrap = """
import runpy
import sys
from irec.models.mclsr import MCLSRModel
original_forward = MCLSRModel.forward
def corrupted_forward(self, inputs):
    outputs = original_forward(self, inputs)
    if self.training:
        outputs['combined_representation'] = outputs['combined_representation'] + 0.1
    return outputs
MCLSRModel.forward = corrupted_forward
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        command[1:1] = ['-c', bootstrap]
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def tiny_configs(tmp_path):
    data_dir = tmp_path / 'data' / 'Tiny'
    data_dir.mkdir(parents=True)
    samples = {
        'train_mclsr': '1 1 2 3 4\n2 2 4 5 6\n3 1 3 6\n',
        'train_sasrec': '1 1 2 3 4\n2 2 4 5 6\n3 1 3 6\n',
        'valid_history': '4 1 2\n',
        'valid_target': '4 5\n',
        'test_history': '5 2 3\n',
        'test_target': '5 6\n',
    }
    for name, content in samples.items():
        (data_dir / f'{name}.txt').write_text(content)
    counts_path = data_dir / 'item_counts.pkl'
    with counts_path.open('wb') as output:
        pickle.dump(np.arange(1, 9, dtype=np.float32), output)
    configs = {}
    for name, filename in (
        ('mclsr', '05_full_l00.json'),
        ('sasrec', 'sasrec_inbatch_l00.json'),
    ):
        config = json.loads((ROOT / 'configs' / 'train' / 'clothing64' / filename).read_text())
        dataset = config['dataset']
        if name == 'mclsr':
            dataset['graph_dir_path'] = str(data_dir)
            dataset = dataset['dataset']
        dataset.update(path_to_data_dir=str(data_dir.parent), name=data_dir.name)
        config['dataloader']['train'].update(batch_size=3, shuffle=False, drop_last=False, pin_memory=False)
        config['model'].update(embedding_dim=8, eval_top_k=3)
        if name == 'sasrec':
            config['model'].update(dim_feedforward=16, num_layers=1)
        config['loss']['losses'][0]['path_to_item_counts'] = str(counts_path)
        configs[name] = config
    return configs, data_dir


@pytest.mark.parametrize('model_name', ['mclsr', 'sasrec'])
@pytest.mark.parametrize('mismatch', [False, True])
def test_architecture_cli_reports_required_check_status(tmp_path, tiny_configs, model_name, mismatch):
    configs, data_dir = tiny_configs
    if mismatch and model_name == 'sasrec':
        other_dir = data_dir.parent / 'Other'
        other_dir.mkdir()
        for filename in data_dir.glob('*.txt'):
            (other_dir / filename.name).write_text(filename.read_text())
        (other_dir / 'test_target.txt').write_text('5 4\n')
        configs['mclsr']['dataset'] = copy.deepcopy(configs['mclsr']['dataset']['dataset'])
        configs['mclsr']['dataset']['name'] = other_dir.name
    for name, config in configs.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(config))
    if model_name == 'mclsr':
        result = _run(
            'indep_check_architecture.py',
            '--config',
            tmp_path / 'mclsr.json',
            '--data',
            data_dir,
            cwd=tmp_path,
            inject_mclsr_mismatch=mismatch,
        )
    else:
        result = _run(
            'indep_check_sasrec.py',
            '--config',
            tmp_path / 'sasrec.json',
            '--mclsr-config',
            tmp_path / 'mclsr.json',
            cwd=tmp_path,
        )
    output = result.stdout + result.stderr
    assert '==== SUMMARY ====' in result.stdout, output
    if mismatch:
        assert 'FAIL' in result.stdout, output
        assert result.returncode == 1, output
    else:
        assert 'FAIL' not in result.stdout, output
        assert result.returncode == 0, output


def _graph_artifacts(tmp_path):
    data_dir = tmp_path / 'data' / 'Tiny'
    data_dir.mkdir(parents=True)
    (data_dir / 'train_mclsr.txt').write_text('1 1 2\n2 1 2\n')
    for filename in ('valid_history.txt', 'valid_target.txt', 'test_history.txt', 'test_target.txt'):
        (data_dir / filename).write_text('')
    general = np.zeros((8, 8))
    general[1:3, 5:7] = 0.5
    general[5:7, 1:3] = 0.5
    similarity = np.zeros((4, 4))
    similarity[1, 2] = similarity[2, 1] = 1.0
    for name, matrix in (
        ('general_graph_trainOnly.npz', general),
        ('user_graph_k50_trainOnly.npz', similarity),
        ('item_graph_k50_trainOnly.npz', similarity),
    ):
        sp.save_npz(data_dir / name, sp.csr_matrix(matrix))
    return data_dir


def test_graph_cli_accepts_valid_artifacts(tmp_path):
    _graph_artifacts(tmp_path)
    result = _run('verify_graphs.py', 'Tiny', cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('graph_name', ['general_graph', 'user_graph_k50', 'item_graph_k50'])
@pytest.mark.parametrize('corruption', ['weight', 'edge', 'nan'])
def test_graph_cli_rejects_invalid_artifacts(tmp_path, graph_name, corruption):
    data_dir = _graph_artifacts(tmp_path)
    path = data_dir / f'{graph_name}_trainOnly.npz'
    graph = sp.load_npz(path).tolil()
    if corruption == 'edge':
        graph[0, 1] = 1.0
    else:
        column = 5 if graph_name == 'general_graph' else 2
        graph[1, column] = float('nan') if corruption == 'nan' else 0.25
    sp.save_npz(path, graph.tocsr())
    result = _run('verify_graphs.py', 'Tiny', cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'FAIL' in result.stdout, result.stdout + result.stderr


def test_graph_cli_requires_a_dataset(tmp_path):
    result = _run('verify_graphs.py', cwd=tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr


def test_direct_logq_runner_includes_paper_oracle(tmp_path):
    result = _run(ROOT / 'tests' / 'test_logq_losses.py', cwd=tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'test_fps_logq_paper_scheme_matches_manual_and_fps: OK' in result.stdout, output


def test_general_only_graph_uses_all_split_catalogue(tmp_path):
    data_dir = _graph_artifacts(tmp_path)
    (data_dir / 'user_graph_k50_trainOnly.npz').unlink()
    (data_dir / 'item_graph_k50_trainOnly.npz').unlink()
    for filename, content in {
        'valid_history.txt': '5 7\n',
        'valid_target.txt': '5 6\n',
        'test_history.txt': '9 3\n',
        'test_target.txt': '9 8\n',
    }.items():
        (data_dir / filename).write_text(content)
    general = np.zeros((21, 21))
    general[1:3, 12:14] = 0.5
    general[12:14, 1:3] = 0.5
    sp.save_npz(data_dir / 'general_graph_trainOnly.npz', sp.csr_matrix(general))
    result = _run('verify_graphs.py', 'Tiny', cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FAIL' not in result.stdout
    assert 'SKIP user-user' in result.stdout


@pytest.mark.parametrize('alpha', [0.5, 0.8])
@pytest.mark.parametrize('shared_projector', [False, True])
@pytest.mark.parametrize('layernorm', [False, True])
def test_architecture_supports_configured_alpha_projectors_and_layernorm(
    tmp_path, tiny_configs, alpha, shared_projector, layernorm
):
    configs, data_dir = tiny_configs
    config = configs['mclsr']
    config['model'].update(
        alpha=alpha,
        share_il_projector=shared_projector,
        use_layernorm=layernorm,
        layer_norm_eps=0.1,
        initializer_range=0.1,
    )
    path = tmp_path / 'configured.json'
    path.write_text(json.dumps(config))
    result = _run('indep_check_architecture.py', '--config', path, '--data', data_dir, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FAIL' not in result.stdout


def test_architecture_identifies_unsupported_graph_layer_mean(tmp_path, tiny_configs):
    configs, data_dir = tiny_configs
    config = configs['mclsr']
    config['model']['graph_layer_mean'] = True
    path = tmp_path / 'unsupported.json'
    path.write_text(json.dumps(config))
    result = _run('indep_check_architecture.py', '--config', path, '--data', data_dir, cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Unsupported verifier configuration' in result.stderr


def test_graph_legacy_catalogue_uses_cropped_history_ids(tmp_path):
    data_dir = _graph_artifacts(tmp_path)
    (data_dir / 'user_graph_k50_trainOnly.npz').unlink()
    (data_dir / 'item_graph_k50_trainOnly.npz').unlink()
    suffix = ' '.join(['1', '2'] * 10)
    (data_dir / 'train_mclsr.txt').write_text(f'1 90 {suffix}\n2 1 2\n')
    (data_dir / 'valid_history.txt').write_text(f'2 91 {suffix}\n')
    (data_dir / 'test_history.txt').write_text(f'2 92 {suffix}\n')
    result = _run('verify_graphs.py', 'Tiny', cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('script', ['verify_graphs.py', 'indep_check_architecture.py'])
def test_legacy_verifiers_explicitly_reject_v2_manifest(tmp_path, tiny_configs, script):
    configs, data_dir = tiny_configs
    (data_dir / 'preprocessing.json').write_text(json.dumps({'protocol': 'windowed-v2'}))
    path = tmp_path / 'mclsr.json'
    path.write_text(json.dumps(configs['mclsr']))
    args = ['Tiny'] if script == 'verify_graphs.py' else ['--config', path, '--data', data_dir]
    result = _run(script, *args, cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Unsupported verifier dataset' in result.stderr
