"""Fail-fast v2 input contracts; legacy behavior and legitimate repeated events remain controls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sparse
import torch

import irec.dataset.graph as graph_module
from irec.dataset.graph import GraphDataset
from irec.dataset.sequences import MCLSRDataset, SASRecDataset


SPEC = importlib.util.spec_from_file_location(
    'prepare_input_contract', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_dataset.py'
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.fixture(autouse=True)
def cpu_graphs(monkeypatch):
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _prepare(directory, history_length=2):
    PREPARE.write_files(
        [(1, [1, 2, 1, 1, 3]), (2, [3, 2, 3, 1]), (3, [1, 2, 1, 2, 3, 1, 2, 1, 1, 1]), (4, [3] * 5)],
        {1, 2},
        {3},
        {4},
        directory,
        protocol='windowed-v2',
        max_history_length=history_length,
    )
    return {
        'path_to_data_dir': str(directory.parent),
        'name': directory.name,
        'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
        'train_sampler': {'type': 'next_item_prediction', 'negative_sampler_type': 'random', 'num_negatives_train': 0},
        'eval_sampler': {'type': 'mclsr'},
    }


def _load(kind, config, cache_directory):
    if kind == 'sasrec':
        return SASRecDataset.create_from_config(config)
    dataset = MCLSRDataset.create_from_config(config)
    if kind == 'mclsr':
        return dataset
    cache_directory.mkdir(exist_ok=True)
    return GraphDataset(dataset, cache_directory)


def _resign(directory):
    path = directory / 'preprocessing.json'
    manifest = json.loads(path.read_text())
    for filename in manifest['files_sha256']:
        manifest['files_sha256'][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def _damage(directory, damage):
    filename = 'train_mclsr.txt'
    if damage.startswith('canonical_'):
        filename = 'train_sasrec.txt'
    elif damage.startswith('history_'):
        filename = 'valid_history.txt'
    elif damage.startswith('target_'):
        filename = 'valid_target.txt'
    path = directory / filename
    rows = [line.split() for line in path.read_text().splitlines()]
    if damage in ['missing_window', 'canonical_missing_user']:
        rows.pop(0)
    elif damage in ['extra_window', 'canonical_duplicate_user', 'history_duplicate_user', 'target_duplicate_user']:
        rows.append(rows[0].copy())
    elif damage == 'overlong_window':
        rows[-1].insert(1, '1')
    elif damage == 'truncated_window':
        rows[-1].pop(1)
    elif damage in ['padding_window', 'canonical_padding', 'history_padding', 'target_padding']:
        rows[0][1] = '0'
    elif damage == 'negative_item':
        rows[0][1] = '-1'
    elif damage == 'mask_item':
        rows[0][1] = '4'
    elif damage == 'padding_user':
        rows[0][0] = '0'
    elif damage == 'blank_window':
        rows.append([])
    elif damage in ['canonical_empty_user', 'history_empty_user', 'target_empty_user']:
        rows[0] = rows[0][:1]
    elif damage == 'history_orphan':
        rows.append(['1', '1', '2'])
    elif damage == 'target_orphan':
        rows.append(['1', '1'])
    elif damage == 'history_bad_80_20':
        rows[0] = rows[0][:4]
    elif damage == 'canonical_missing_event':
        rows[0].pop()
    else:
        raise AssertionError(damage)
    path.write_text('\n'.join(' '.join(row) for row in rows) + '\n')
    _resign(directory)


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'binary-graph'])
@pytest.mark.parametrize(
    'damage',
    [
        'missing_window',
        'extra_window',
        'overlong_window',
        'truncated_window',
        'padding_window',
        'negative_item',
        'mask_item',
        'padding_user',
        'blank_window',
        'canonical_duplicate_user',
        'canonical_missing_user',
        'canonical_missing_event',
        'canonical_empty_user',
        'canonical_padding',
        'history_duplicate_user',
        'target_duplicate_user',
        'history_orphan',
        'target_orphan',
        'history_empty_user',
        'target_empty_user',
        'history_padding',
        'target_padding',
        'history_bad_80_20',
    ],
)
def test_v2_rejects_invalid_raw_rows_even_with_matching_file_hashes(tmp_path, kind, damage):
    directory = tmp_path / 'data'
    config = _prepare(directory)
    _damage(directory, damage)
    with pytest.raises(ValueError, match='canonical|window|non-padding|evaluation|80/20'):
        _load(kind, config, tmp_path / 'cache')
    assert not list(tmp_path.glob('cache/*.npz'))


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'binary-graph'])
@pytest.mark.parametrize('overlap', ['train-valid', 'valid-test'])
def test_v2_rejects_cross_split_user_aliases(tmp_path, kind, overlap):
    directory = tmp_path / 'data'
    config = _prepare(directory)
    split, replacement = ('valid', '1') if overlap == 'train-valid' else ('test', '3')
    for suffix in ['history', 'target']:
        path = directory / f'{split}_{suffix}.txt'
        _, *items = path.read_text().split()
        path.write_text(' '.join([replacement, *items]) + '\n')
    _resign(directory)
    with pytest.raises(ValueError, match='canonical|evaluation|unique'):
        _load(kind, config, tmp_path / 'cache')


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec'])
@pytest.mark.parametrize('field', ['max_history_length', 'max_sequence_length', 'eval_history_max_len'])
@pytest.mark.parametrize('value', [True, 1.0, 0, -1, '1'])
def test_v2_rejects_nonpositive_or_noninteger_cap_before_cropping(tmp_path, kind, field, value):
    config = _prepare(tmp_path / 'data', history_length=1)
    config[field] = value
    with pytest.raises(ValueError, match='history|length'):
        _load(kind, config, tmp_path / 'cache')


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec'])
@pytest.mark.parametrize('manifest_value', [[], None, {'protocol': 'windowed-v2', 'max_history_length': True}])
def test_invalid_manifest_shape_and_boolean_cap_raise_actionable_value_error(tmp_path, kind, manifest_value):
    config = _prepare(tmp_path / 'data', history_length=1)
    if isinstance(manifest_value, dict):
        manifest = json.loads((tmp_path / 'data/preprocessing.json').read_text())
        manifest.update(manifest_value)
        manifest_value = manifest
    (tmp_path / 'data/preprocessing.json').write_text(json.dumps(manifest_value))
    with pytest.raises(ValueError, match='manifest'):
        _load(kind, config, tmp_path / 'cache')


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'binary-graph'])
def test_valid_repeated_windows_reordering_and_heldout_overlap_are_not_rejected(tmp_path, kind):
    directory = tmp_path / 'data'
    config = _prepare(directory, history_length=1)
    path = directory / 'train_mclsr.txt'
    original = path.read_text().splitlines()
    path.write_text('\n'.join(reversed(original)) + '\n')
    _resign(directory)
    dataset = _load(kind, config, tmp_path / 'cache')
    _, validation, test = dataset.get_samplers()
    assert validation[0]['item.ids'] == [1]
    assert validation[0]['labels.ids'] == [1, 1]
    assert test[0]['labels.ids'] == [3]


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
@pytest.mark.parametrize('mutation', ['file', 'loaded-window'])
def test_graph_revalidates_canonical_source_even_in_binary_default(tmp_path, mode, mutation):
    directory = tmp_path / 'data'
    config = _prepare(directory)
    dataset = MCLSRDataset.create_from_config(config)
    if mutation == 'file':
        path = directory / 'train_sasrec.txt'
        path.write_text(path.read_text() + '1 1 1\n')
    else:
        dataset.get_samplers()[0].dataset[-1]['item.ids'].append(1)
    with pytest.raises(ValueError, match='hash|window'):
        GraphDataset(dataset, tmp_path, interaction_weighting=mode)
    assert not list(tmp_path.glob('*.npz'))


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2', float('nan')])
@pytest.mark.parametrize('entry', ['constructor', 'filter'])
def test_invalid_neighborhood_size_fails_before_cache_writes(tmp_path, value, entry):
    with pytest.raises(ValueError, match='neighborhood|top.k|positive integer'):
        if entry == 'filter':
            GraphDataset._filter_matrix_by_top_k(sparse.csr_matrix([[0.0, 2.0], [2.0, 0.0]]), value)
        else:
            dataset = MCLSRDataset.create_from_config(_prepare(tmp_path / 'data'))
            GraphDataset(dataset, tmp_path, use_user_graph=True, neighborhood_size=value)
    assert not list(tmp_path.glob('*.npz'))


def test_mclsr_with_heldout_graph_inputs_reports_unsupported_raw_sample_schema(tmp_path):
    dataset = MCLSRDataset.create_from_config(_prepare(tmp_path / 'data'))
    with pytest.raises(ValueError, match='item.ids|evaluation|unsupported'):
        GraphDataset(dataset, tmp_path, use_train_data_only=False)
    assert not list(tmp_path.glob('*.npz'))


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_empty_training_partition_has_finite_zero_graphs(tmp_path, mode):
    directory = tmp_path / 'data'
    PREPARE.write_files(
        [(1, [1] * 5), (2, [2] * 5)],
        set(),
        {1},
        {2},
        directory,
        protocol='windowed-v2',
        max_history_length=2,
    )
    config = {
        'path_to_data_dir': str(tmp_path),
        'name': 'data',
        'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
    }
    dataset = MCLSRDataset.create_from_config(config)
    graph = GraphDataset(dataset, tmp_path, use_user_graph=True, use_item_graph=True, interaction_weighting=mode)
    for key in ['graph', 'user_graph', 'item_graph']:
        assert graph.meta[key]._nnz() == 0
        assert torch.isfinite(graph.meta[key].to_dense()).all()


def test_legacy_loader_retains_historical_cropping_and_intersection_behavior(tmp_path):
    directory = tmp_path / 'legacy'
    config = _prepare(directory)
    (directory / 'preprocessing.json').unlink()
    (directory / 'train_mclsr.txt').write_text('0 0 1 2 3\n')
    (directory / 'valid_history.txt').write_text('3 1 2 3\n3 2 1 2\n99 1 2\n')
    config['max_sequence_length'] = 2
    dataset = MCLSRDataset.create_from_config(config)
    train, validation, _ = dataset.get_samplers()
    assert train[0]['user.ids'] == [0]
    assert train[0]['item.ids'] == [2]
    assert train[0]['labels.ids'] == [3]
    assert len(validation) == 1
    assert validation[0]['item.ids'] == [1, 2]
    with pytest.raises(ValueError, match='windowed-v2'):
        dataset.get_train_event_sequences()


def test_top_k_larger_than_catalogue_is_a_valid_noop():
    matrix = sparse.csr_matrix([[0.0, 2.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(GraphDataset._filter_matrix_by_top_k(matrix, 100).toarray(), matrix.toarray())


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'binary-graph'])
def test_valid_v2_long_histories_and_unused_catalogue_ids_preserve_capacity(tmp_path, kind):
    directory = tmp_path / 'data'
    PREPARE.write_files(
        [(1, [1, 2, 3] * 9), (2, [1, 2] * 15), (3, [3] * 5)],
        {1},
        {2},
        {3},
        directory,
        protocol='windowed-v2',
        max_history_length=20,
    )
    manifest_path = directory / 'preprocessing.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(num_users=8, num_items=9)
    manifest_path.write_text(json.dumps(manifest))
    config = {
        'path_to_data_dir': str(tmp_path),
        'name': 'data',
        'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
        'train_sampler': {'type': 'next_item_prediction', 'negative_sampler_type': 'random', 'num_negatives_train': 0},
        'eval_sampler': {'type': 'mclsr'},
    }
    dataset = _load(kind, config, tmp_path / 'cache')
    train, validation, _ = dataset.get_samplers()
    assert train[len(train) - 1]['item.length'] == 20
    assert validation[0]['item.ids'] == [1, 2] * 10
    assert validation[0]['labels.ids'] == [1, 2] * 3
    assert dataset.meta['num_users'] == 8
    assert dataset.meta['num_items'] == 9


def test_v2_sasrec_cannot_bypass_canonical_sources_with_custom_train_part(tmp_path):
    config = _prepare(tmp_path / 'data')
    (tmp_path / 'data/custom.txt').write_text('1 3 3 3 3\n')
    config['train_part'] = 'custom'
    with pytest.raises(ValueError, match='train_part|canonical'):
        SASRecDataset.create_from_config(config)


def test_numpy_integer_top_k_retains_existing_valid_direct_call_contract(tmp_path):
    matrix = sparse.csr_matrix([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_array_equal(GraphDataset._filter_matrix_by_top_k(matrix, np.int64(1)).toarray(), matrix.toarray())
    dataset = MCLSRDataset.create_from_config(_prepare(tmp_path / 'data'))
    graph = GraphDataset(dataset, tmp_path, use_user_graph=True, neighborhood_size=np.int64(1))
    assert graph.meta['user_graph'] is not None
