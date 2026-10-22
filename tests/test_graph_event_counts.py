"""Graph weights come from canonical events, never overlapping window occurrences."""

from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from irec.dataset.base import BaseDataset
from irec.dataset.graph import GraphDataset
import irec.dataset.graph as graph_module
from irec.dataset.sequences import MCLSRDataset


SPEC = importlib.util.spec_from_file_location(
    'prepare_event_graph_fixture', Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_dataset.py'
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.fixture(autouse=True)
def cpu_graphs(monkeypatch):
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))


def _prepare(directory, sequences=None, history_length=2):
    if sequences is None:
        sequences = [(1, [1, 2, 1, 1, 3]), (2, [1, 2, 2, 3]), (3, [2, 3, 3])]
    all_sequences = sequences + [(8, [4] * 5), (9, [4] * 5)]
    PREPARE.write_files(
        all_sequences,
        {user for user, _ in sequences},
        {8},
        {9},
        directory,
        protocol='windowed-v2',
        max_history_length=history_length,
    )
    return {
        'type': 'mclsr',
        'path_to_data_dir': str(directory.parent),
        'name': directory.name,
        'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
    }


def _build(config, cache_dir, mode='event_count', **options):
    cache_dir.mkdir(exist_ok=True, parents=True)
    return BaseDataset.create_from_config(
        {
            'type': 'graph',
            'dataset': config,
            'graph_dir_path': str(cache_dir),
            'use_user_graph': True,
            'use_item_graph': True,
            'interaction_weighting': mode,
            **options,
        }
    )


def _normalize(adjacency):
    degrees = adjacency.sum(axis=1)
    scale = np.zeros_like(degrees)
    np.divide(1, np.sqrt(degrees), out=scale, where=degrees > 0)
    return adjacency * scale[:, None] * scale[None, :]


def _oracles(sequences, num_users, num_items, mode):
    incidence = np.zeros((num_users + 2, num_items + 2), dtype=float)
    for user_id, items in sequences:
        for item_id in items:
            incidence[user_id, item_id] += 1
    if mode == 'binary':
        incidence = (incidence > 0).astype(float)
    bipartite = np.block(
        [[np.zeros((num_users + 2, num_users + 2)), incidence], [incidence.T, np.zeros((num_items + 2, num_items + 2))]]
    )
    users = incidence @ incidence.T
    items = incidence.T @ incidence
    np.fill_diagonal(users, 0)
    np.fill_diagonal(items, 0)
    return {'graph': bipartite, 'user_graph': users, 'item_graph': items}


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
@pytest.mark.parametrize('history_length', [1, 2, 20])
def test_repeated_events_match_dense_oracle_not_window_counts(tmp_path, mode, history_length):
    sequences = [(1, [1, 2, 1, 1, 3]), (2, [1, 2, 2, 3]), (3, [2, 3, 3])]
    config = _prepare(tmp_path / 'data', sequences, history_length)
    graph = _build(config, tmp_path / 'cache', mode)
    for key, raw in _oracles(sequences, 5, 4, mode).items():
        np.testing.assert_allclose(graph.meta[key].to_dense().numpy(), _normalize(raw), atol=1e-7)
    expected_counts = Counter((user, item) for user, items in sequences for item in items)
    windows = graph.get_samplers()[0].dataset
    window_counts = Counter((row['user.ids'][0], item) for row in windows for item in row['item.ids'])
    assert expected_counts != window_counts


def test_event_graphs_equal_binary_without_repeated_events(tmp_path):
    config = _prepare(tmp_path / 'data', [(1, [1, 2, 3]), (2, [2, 3, 4]), (3, [1, 3, 4])])
    binary = _build(config, tmp_path / 'cache', 'binary')
    events = _build(config, tmp_path / 'cache', 'event_count')
    for key in ('graph', 'user_graph', 'item_graph'):
        torch.testing.assert_close(events.meta[key].to_dense(), binary.meta[key].to_dense())


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_raw_weights_match_exact_incidence_and_coaction_products(tmp_path, monkeypatch, mode):
    sequences = [(1, [1, 2, 1, 1, 3]), (2, [1, 2, 2, 3]), (3, [2, 3, 3])]
    config = _prepare(tmp_path / 'data', sequences)
    captured = []
    normalize = GraphDataset.get_sparse_graph_layer

    def capture(matrix, first_dim, second_dim, biparite=False):
        captured.append(matrix.toarray())
        return normalize(matrix, first_dim, second_dim, biparite=biparite)

    monkeypatch.setattr(GraphDataset, 'get_sparse_graph_layer', staticmethod(capture))
    _build(config, tmp_path / 'cache', mode)
    expected = _oracles(sequences, 5, 4, mode)
    np.testing.assert_array_equal(captured[0], expected['graph'][:7, 7:])
    np.testing.assert_array_equal(captured[1], expected['user_graph'])
    np.testing.assert_array_equal(captured[2], expected['item_graph'])


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_no_coactions_and_reserved_nodes_stay_zero(tmp_path, mode):
    sequences = [(1, [1, 1, 1]), (2, [2, 2, 2]), (3, [3, 3, 3])]
    config = _prepare(tmp_path / 'data', sequences)
    actual = _build(config, tmp_path / 'cache', mode)
    for key, raw in _oracles(sequences, 5, 4, mode).items():
        np.testing.assert_allclose(actual.meta[key].to_dense().numpy(), _normalize(raw), atol=1e-7)
    assert actual.meta['user_graph']._nnz() == actual.meta['item_graph']._nnz() == 0


def test_event_count_cache_has_distinct_names_and_semantic_manifest(tmp_path):
    config = _prepare(tmp_path / 'data')
    _build(config, tmp_path / 'cache', 'binary')
    binary_bytes = {file.name: file.read_bytes() for file in (tmp_path / 'cache').glob('*.npz')}
    _build(config, tmp_path / 'cache', 'event_count')
    assert len(list((tmp_path / 'cache').glob('*.npz'))) == 6
    for filename, content in binary_bytes.items():
        assert (tmp_path / 'cache' / filename).read_bytes() == content
    for file in (tmp_path / 'cache').glob('*event_count*.npz'):
        with np.load(file, allow_pickle=False) as payload:
            manifest = json.loads(payload['cache_manifest'].item())
        assert manifest['interaction_weighting'] == 'event_count'


def test_same_pairs_changed_event_multiplicity_invalidates_all_graphs(tmp_path):
    config = _prepare(tmp_path / 'first')
    changed = _prepare(tmp_path / 'changed', [(1, [1, 2, 3]), (2, [1, 2, 2, 3]), (3, [2, 3, 3])])
    original = _build(config, tmp_path / 'shared')
    actual = _build(changed, tmp_path / 'shared')
    expected = _build(changed, tmp_path / 'fresh')
    for key in ('graph', 'user_graph', 'item_graph'):
        assert not torch.allclose(original.meta[key].to_dense(), expected.meta[key].to_dense())
        torch.testing.assert_close(actual.meta[key].to_dense(), expected.meta[key].to_dense())


@pytest.mark.parametrize('reorder', [False, True])
def test_equivalent_events_reuse_cache_despite_different_history_cap(tmp_path, monkeypatch, reorder):
    config = _prepare(tmp_path / 'short', history_length=1)
    changed = _prepare(tmp_path / 'long', history_length=20)
    if reorder:
        for filename in ('train_sasrec.txt', 'train_mclsr.txt'):
            source = tmp_path / 'long' / filename
            source.write_text('\n'.join(reversed(source.read_text().splitlines())) + '\n')
            _resign(source.parent, filename)
    original = _build(config, tmp_path / 'cache')

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail('Same canonical event counts must reuse the graph cache')

    monkeypatch.setattr(GraphDataset, 'get_sparse_graph_layer', unexpected_rebuild)
    actual = _build(changed, tmp_path / 'cache')
    for key in ('graph', 'user_graph', 'item_graph'):
        torch.testing.assert_close(actual.meta[key].to_dense(), original.meta[key].to_dense())


@pytest.mark.parametrize('invalid_mode', ['count', '', None])
def test_unknown_weighting_cannot_silently_use_binary(tmp_path, invalid_mode):
    config = _prepare(tmp_path / 'data')
    with pytest.raises(ValueError, match='interaction_weighting'):
        _build(config, tmp_path / 'cache', invalid_mode)


def test_event_counts_reject_legacy_data_instead_of_guessing_from_windows(tmp_path):
    config = _prepare(tmp_path / 'data')
    (tmp_path / 'data' / 'preprocessing.json').unlink()
    with pytest.raises(ValueError, match='windowed-v2'):
        _build(config, tmp_path / 'cache')


def test_event_counts_require_train_only(tmp_path):
    config = _prepare(tmp_path / 'data')
    with pytest.raises(ValueError, match='train.only'):
        _build(config, tmp_path / 'cache', use_train_data_only=False)


def test_unverified_dataset_cannot_supply_event_weights(tmp_path):
    config = _prepare(tmp_path / 'data')
    dataset = MCLSRDataset.create_from_config(config)
    dataset.get_train_event_sequences = None
    with pytest.raises(ValueError, match='canonical'):
        GraphDataset(dataset, tmp_path, interaction_weighting='event_count')


def _resign(directory, filename):
    manifest_path = directory / 'preprocessing.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files_sha256'][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('damage', ['duplicate_user', 'missing_event', 'padding', 'heldout', 'extra_window'])
def test_event_source_must_be_canonical_and_match_training_windows(tmp_path, damage):
    config = _prepare(tmp_path / 'data')
    filename = 'train_sasrec.txt'
    source = tmp_path / 'data' / filename
    text = source.read_text()
    if damage == 'duplicate_user':
        text += text.splitlines()[0] + '\n'
    elif damage == 'missing_event':
        text = text.replace('1 1 2 1 1 3', '1 1 2 1 3')
    elif damage == 'padding':
        text = text.replace('1 1 2 1 1 3', '1 0 2 1 1 3')
    elif damage == 'heldout':
        text += '4 4 4 4 4 4\n'
    else:
        filename = 'train_mclsr.txt'
        source = tmp_path / 'data' / filename
        text = source.read_text() + source.read_text().splitlines()[0] + '\n'
    source.write_text(text)
    _resign(tmp_path / 'data', filename)
    with pytest.raises(ValueError, match='event|canonical|window'):
        _build(config, tmp_path / 'cache')


def test_event_source_mutation_after_dataset_loading_is_rejected(tmp_path):
    config = _prepare(tmp_path / 'data')
    dataset = MCLSRDataset.create_from_config(config)
    source = tmp_path / 'data' / 'train_sasrec.txt'
    source.write_text(source.read_text().replace('1 1 2 1 1 3', '1 1 2 1 3'))
    with pytest.raises(ValueError, match='hash'):
        GraphDataset(dataset, tmp_path, interaction_weighting='event_count')


@pytest.mark.parametrize('extra_item', [2, 999])
def test_raw_overlong_window_cannot_be_hidden_by_loader_crop(tmp_path, extra_item):
    config = _prepare(tmp_path / 'data')
    source = tmp_path / 'data' / 'train_mclsr.txt'
    lines = source.read_text().splitlines()
    user, *items = lines[-1].split()
    assert len(items) == 3
    lines[-1] = ' '.join([user, str(extra_item), *items])
    source.write_text('\n'.join(lines) + '\n')
    _resign(source.parent, source.name)
    with pytest.raises(ValueError, match='event|window'):
        _build(config, tmp_path / 'cache')


@pytest.mark.parametrize('filename', ['valid_history.txt', 'valid_target.txt', 'test_history.txt', 'test_target.txt'])
def test_raw_orphan_heldout_user_cannot_be_hidden_by_eval_intersection(tmp_path, filename):
    config = _prepare(tmp_path / 'data')
    source = tmp_path / 'data' / filename
    source.write_text(source.read_text() + '1 1 2\n')
    _resign(source.parent, source.name)
    with pytest.raises(ValueError, match='held.out|evaluation|canonical'):
        _build(config, tmp_path / 'cache')


def test_weighted_top_k_retains_strongest_coactions(tmp_path):
    sequences = [(1, [1, 1, 1, 1, 1, 2, 3]), (2, [1, 1, 2, 2, 3]), (3, [2, 3, 3, 3])]
    config = _prepare(tmp_path / 'data', sequences)
    actual = _build(config, tmp_path / 'cache', neighborhood_size=1)
    for key, raw in _oracles(sequences, 5, 4, 'event_count').items():
        if key == 'graph':
            continue
        kept = np.zeros_like(raw)
        for row in range(len(raw)):
            if raw[row].max() > 0:
                column = raw[row].argmax()
                assert np.count_nonzero(raw[row] == raw[row, column]) == 1
                kept[row, column] = raw[row, column]
        np.testing.assert_allclose(actual.meta[key].to_dense().numpy(), _normalize(kept), atol=1e-7)


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
def test_all_length_four_sequences_over_two_items(tmp_path, mode):
    for index, sequence in enumerate(itertools.product((1, 2), repeat=4)):
        sequences = [(1, list(sequence)), (2, [1, 2, 2]), (3, [2, 3])]
        config = _prepare(tmp_path / f'data-{index}', sequences, history_length=1 + index % 3)
        actual = _build(config, tmp_path / f'cache-{index}', mode)
        for key, raw in _oracles(sequences, 5, 4, mode).items():
            np.testing.assert_allclose(actual.meta[key].to_dense().numpy(), _normalize(raw), atol=1e-7)
