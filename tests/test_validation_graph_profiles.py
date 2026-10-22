import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from irec.dataset import BaseDataset
import irec.dataset.graph as graph_module
from irec.dataset.sequences import _sequence_lengths


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = ROOT / 'configs/train/validation'


def _profile(mode):
    path = PROFILE_DIRECTORY / f'clothing_v2_{mode}_l00.json'
    assert path.is_file(), f'missing explicit graph profile: {path}'
    return json.loads(path.read_text())


def test_graph_comparison_changes_only_weighting_and_run_name():
    baseline = _profile('binary')
    frequency = _profile('event_count')
    assert baseline['experiment_name'] != frequency['experiment_name']
    assert baseline['dataset']['interaction_weighting'] == 'binary'
    assert frequency['dataset']['interaction_weighting'] == 'event_count'
    expected = copy.deepcopy(baseline)
    expected['experiment_name'] = frequency['experiment_name']
    expected['dataset']['interaction_weighting'] = 'event_count'
    assert frequency == expected


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_graph_profiles_use_v2_and_keep_logq_disabled(mode):
    config = _profile(mode)
    wrapper = config['dataset']
    dataset = wrapper['dataset']
    assert wrapper['use_train_data_only'] is True
    assert wrapper['use_user_graph'] is wrapper['use_item_graph'] is True
    assert wrapper['neighborhood_size'] == 50
    assert dataset['name'] == 'Clothing_v2'
    assert dataset['expected_protocol'] == 'windowed-v2'
    assert dataset['max_history_length'] == dataset['max_sequence_length'] == 20
    assert Path(wrapper['graph_dir_path']) == Path(dataset['path_to_data_dir']) / dataset['name']
    assert dataset['samplers']['num_negatives_train'] == 0
    downstream, *auxiliaries = config['loss']['losses']
    assert downstream['type'] == 'mclsr_logq_inbatch'
    assert downstream['logq_lambda'] == 0
    assert downstream['expected_counts_role'] == 'target'
    assert downstream['expected_counts_max_len'] == 21
    assert Path(downstream['path_to_item_counts']).parent == Path(wrapper['graph_dir_path'])
    assert [loss['weight'] for loss in config['loss']['losses']] == [1.0, 1.0, 0.05, 0.05]
    assert all(loss['type'] == 'fps' for loss in auxiliaries)
    assert all(loss['scheme'] == 'paper' and loss['normalize_embeddings'] for loss in auxiliaries)
    assert all(loss['temperature'] == 0.5 for loss in auxiliaries)
    assert config['model']['share_il_projector'] is True
    assert config['best_metric'] == 'validation/ndcg@20'
    assert not any(callback['type'] == 'eval' for callback in config['callback']['callbacks'])
    assert config['seed'] == 1
    assert config['train_steps_num'] == 65536
    assert config['epochs_threshold'] == config['train_steps_num']


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_graph_profiles_reject_missing_v2_manifest(tmp_path, mode):
    config = _profile(mode)['dataset']['dataset']
    with pytest.raises(ValueError, match='requires a windowed-v2 preprocessing manifest'):
        _sequence_lengths(config, tmp_path)


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_graph_profiles_reject_legacy_manifest(tmp_path, mode):
    (tmp_path / 'preprocessing.json').write_text(json.dumps({'protocol': 'legacy'}))
    config = _profile(mode)['dataset']['dataset']
    with pytest.raises(ValueError, match='unsupported preprocessing protocol'):
        _sequence_lengths(config, tmp_path)


@pytest.mark.parametrize('expected_protocol', ['', 'legacy', True])
def test_required_protocol_rejects_unsupported_configuration(tmp_path, expected_protocol):
    with pytest.raises(ValueError, match='expected_protocol must be windowed-v2'):
        _sequence_lengths({'expected_protocol': expected_protocol}, tmp_path)


def test_unpinned_legacy_config_still_accepts_missing_manifest(tmp_path):
    assert _sequence_lengths({'max_sequence_length': 20}, tmp_path) == (20, 20, 20, {})
    assert _sequence_lengths({'max_history_length': 20}, tmp_path) == (21, 20, 20, {})


@pytest.mark.parametrize('mode,expected_frequency', [('binary', 1), ('event_count', 2)])
def test_new_profile_constructs_graphs_from_canonical_repeated_events(tmp_path, monkeypatch, mode, expected_frequency):
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))
    specification = importlib.util.spec_from_file_location(
        'prepare_graph_profile_fixture', ROOT / 'scripts/prepare_dataset.py'
    )
    preparation = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(preparation)
    config = _profile(mode)['dataset']
    directory = tmp_path / 'Clothing_v2'
    preparation.write_files(
        [(1, [1, 1, 2, 3, 4]), (2, [1, 2, 2, 3, 4]), (3, [1, 2, 3, 4, 1]), (4, [2, 3, 4, 1, 2])],
        {1, 2},
        {3},
        {4},
        directory,
        protocol='windowed-v2',
        max_history_length=20,
    )
    config['dataset']['path_to_data_dir'] = str(tmp_path)
    config['graph_dir_path'] = str(directory)
    dataset = BaseDataset.create_from_config(config)
    assert dataset._interaction_counts[(1, 1)] == expected_frequency
    for key in ('graph', 'user_graph', 'item_graph'):
        assert dataset.meta[key]._nnz() > 0
        assert torch.isfinite(dataset.meta[key].coalesce().values()).all()
