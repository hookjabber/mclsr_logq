"""Numeric reference tests for the graph branch: adjacency normalisation, similarity-graph
construction and LightGCN propagation are compared with hand-computed dense results."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from irec.dataset.graph import GraphDataset  # noqa: E402
import irec.dataset.graph as graph_module


_GRAPH_CACHE_FILES = [
    ('graph', 'general_graph_trainOnly.npz'),
    ('user_graph', 'user_graph_full_trainOnly.npz'),
    ('item_graph', 'item_graph_full_trainOnly.npz'),
]


@pytest.fixture(autouse=True)
def _cpu_graphs(monkeypatch):
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))


def _cached_dataset(cache_dir, train=None, validation=None, test=None, num_users=3, num_items=4, **options):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if train is None:
        train = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    samplers = tuple(
        SimpleNamespace(dataset=[{'user.ids': [user_id], 'item.ids': item_ids} for user_id, item_ids in split])
        for split in (train, validation or [], test or [])
    )
    dataset = SimpleNamespace(num_users=num_users, num_items=num_items, meta={}, get_samplers=lambda: samplers)
    return GraphDataset(dataset, cache_dir, use_user_graph=True, use_item_graph=True, **options)


@pytest.mark.parametrize('graph_name', ['graph', 'user_graph', 'item_graph'])
def test_graph_cache_invalidates_same_shape_changed_interactions(tmp_path, graph_name):
    original = _cached_dataset(tmp_path / 'shared').meta[graph_name].to_dense()
    changed = [(1, [1, 3]), (2, [2, 3]), (3, [3, 4])]
    reloaded = _cached_dataset(tmp_path / 'shared', train=changed).meta[graph_name].to_dense()
    expected = _cached_dataset(tmp_path / 'fresh', train=changed).meta[graph_name].to_dense()

    assert original.shape == expected.shape
    assert not torch.allclose(original, expected)
    torch.testing.assert_close(reloaded, expected)


@pytest.mark.parametrize('num_users,num_items', [(4, 4), (3, 5), (4, 3)])
def test_graph_cache_invalidates_catalogue_dimensions(tmp_path, num_users, num_items):
    train = [(1, [1, 2]), (2, [2, 3])]
    _cached_dataset(tmp_path / 'shared', train=train)
    reloaded = _cached_dataset(tmp_path / 'shared', train=train, num_users=num_users, num_items=num_items)
    expected = _cached_dataset(tmp_path / 'fresh', train=train, num_users=num_users, num_items=num_items)

    for graph_name, _ in _GRAPH_CACHE_FILES:
        torch.testing.assert_close(reloaded.meta[graph_name].to_dense(), expected.meta[graph_name].to_dense())


def test_graph_cache_reuses_equivalent_interactions(tmp_path, monkeypatch):
    expected = _cached_dataset(tmp_path)

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail('Equivalent interaction sets should reuse the cache')

    monkeypatch.setattr(GraphDataset, 'get_sparse_graph_layer', unexpected_rebuild)
    reordered = [(3, [4, 3]), (1, [2, 1, 2]), (2, [3, 2]), (1, [1])]
    reloaded = _cached_dataset(tmp_path, train=reordered, validation=[(2, [1, 4])], test=[(3, [1])])

    for graph_name, _ in _GRAPH_CACHE_FILES:
        torch.testing.assert_close(reloaded.meta[graph_name].to_dense(), expected.meta[graph_name].to_dense())


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_graph_cache_hashes_selected_evaluation_interactions(tmp_path, split):
    _cached_dataset(tmp_path / 'shared', use_train_data_only=False)
    options = {'use_train_data_only': False, split: [(1, [3]), (2, [4])]}
    reloaded = _cached_dataset(tmp_path / 'shared', **options)
    expected = _cached_dataset(tmp_path / 'fresh', **options)

    for graph_name, _ in _GRAPH_CACHE_FILES:
        torch.testing.assert_close(reloaded.meta[graph_name].to_dense(), expected.meta[graph_name].to_dense())


def test_graph_cache_invalidates_builder_version(tmp_path, monkeypatch):
    _cached_dataset(tmp_path)
    original_builder = GraphDataset.get_sparse_graph_layer
    rebuilt = []

    def track_rebuild(*args, **kwargs):
        rebuilt.append(True)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(GraphDataset, 'get_sparse_graph_layer', staticmethod(track_rebuild))
    monkeypatch.setattr(graph_module, '_GRAPH_BUILDER_VERSION', 'next-version', raising=False)
    _cached_dataset(tmp_path)

    assert len(rebuilt) == 3


@pytest.mark.parametrize('graph_name,filename', _GRAPH_CACHE_FILES)
@pytest.mark.parametrize('damage', ['legacy', 'truncated', 'manifest', 'shape', 'nonfinite', 'indices'])
def test_graph_cache_rebuilds_untrusted_artifacts(tmp_path, graph_name, filename, damage):
    expected = _cached_dataset(tmp_path).meta[graph_name].to_dense()
    cache_path = tmp_path / filename
    with np.load(cache_path, allow_pickle=False) as cached:
        payload = {key: cached[key] for key in cached.files}

    if damage == 'legacy':
        sp.save_npz(cache_path, sp.eye(expected.shape[0], format='csr'))
    elif damage == 'truncated':
        cache_path.write_bytes(cache_path.read_bytes()[:20])
    else:
        if damage == 'manifest':
            payload['cache_manifest'] = np.array('{}')
            payload['data'] = np.zeros_like(payload['data'])
        elif damage == 'shape':
            payload['shape'] = payload['shape'] + [0, 1]
        elif damage == 'nonfinite':
            payload['data'][:] = np.nan
        elif damage == 'indices':
            payload['indices'][0] = expected.shape[0]
        np.savez_compressed(cache_path, **payload)

    reloaded = _cached_dataset(tmp_path).meta[graph_name].to_dense()
    torch.testing.assert_close(reloaded, expected)
    assert sp.load_npz(cache_path).shape == tuple(expected.shape)


@pytest.mark.parametrize('option', ['neighborhood_size', 'use_train_data_only'])
def test_graph_cache_validates_options_in_manifest(tmp_path, option):
    options = {'neighborhood_size': 1} if option == 'neighborhood_size' else {'use_train_data_only': False}
    expected = _cached_dataset(tmp_path / 'expected', **options)
    _cached_dataset(tmp_path / 'shared')
    if option == 'neighborhood_size':
        for entity_type in ('user', 'item'):
            source = tmp_path / 'shared' / f'{entity_type}_graph_full_trainOnly.npz'
            source.rename(source.with_name(f'{entity_type}_graph_k1_trainOnly.npz'))
    else:
        for source in (tmp_path / 'shared').glob('*.npz'):
            source.rename(source.with_name(source.name.replace('trainOnly', 'withValTest')))

    reloaded = _cached_dataset(tmp_path / 'shared', **options)
    for graph_name, _ in _GRAPH_CACHE_FILES:
        torch.testing.assert_close(reloaded.meta[graph_name].to_dense(), expected.meta[graph_name].to_dense())

    for cache_path in (tmp_path / 'expected').glob('*.npz'):
        with np.load(cache_path, allow_pickle=False) as fresh:
            with np.load(tmp_path / 'shared' / cache_path.name, allow_pickle=False) as cached:
                assert cached['cache_manifest'].item() == fresh['cache_manifest'].item()


def test_graph_cache_interrupted_write_preserves_previous_cache(tmp_path, monkeypatch):
    _cached_dataset(tmp_path)
    before = {cache_path.name: cache_path.read_bytes() for cache_path in tmp_path.iterdir()}
    observed = []

    def interrupt_replace(source, destination):
        assert os.path.dirname(source) == str(tmp_path)
        assert os.path.dirname(destination) == str(tmp_path)
        observed.append(sp.load_npz(source).shape)
        raise OSError('simulated interruption before atomic replace')

    with monkeypatch.context() as patch:
        patch.setattr(graph_module.os, 'replace', interrupt_replace)
        with pytest.raises(OSError, match='simulated interruption'):
            _cached_dataset(tmp_path, train=[(1, [1, 3]), (2, [2, 3]), (3, [3, 4])])

    assert len(observed) == 1
    assert {cache_path.name: cache_path.read_bytes() for cache_path in tmp_path.iterdir()} == before


def _dense_sym_norm(adj):
    d = adj.sum(1)
    with np.errstate(divide='ignore'):
        d_inv = np.where(d > 0, d**-0.5, 0.0)
    return d_inv[:, None] * adj * d_inv[None, :]


def test_bipartite_layer_matches_hand_computed_normalisation():
    # 3 users (+2 pad rows) x 4 items (+2 pad rows); interactions u0:{i0,i1}, u1:{i1}, u2:{i1,i2,i3}
    users = np.array([1, 1, 2, 3, 3, 3])
    items = np.array([1, 2, 2, 2, 3, 4])
    R = sp.csr_matrix((np.ones(6), (users, items)), shape=(5, 6))
    got = GraphDataset.get_sparse_graph_layer(R, 5, 6, biparite=True).toarray()
    adj = np.zeros((11, 11))
    adj[:5, 5:] = R.toarray()
    adj[5:, :5] = R.toarray().T
    assert np.allclose(got, _dense_sym_norm(adj)), 'bipartite normalisation differs from D^-1/2 A D^-1/2'
    assert np.allclose(got, got.T), 'bipartite normalised adjacency must be symmetric'


def test_similarity_graph_weights_are_coaction_counts_and_top_k():
    # user-user co-action counts from u0:{i0,i1}, u1:{i1,i2}, u2:{i0,i1,i2}:
    # (u0,u1)=1 (i1), (u0,u2)=2 (i0,i1), (u1,u2)=2 (i1,i2)
    interactions = {0: {0, 1}, 1: {1, 2}, 2: {0, 1, 2}}
    item_2_users = {0: {0, 2}, 1: {0, 1, 2}, 2: {1, 2}}
    fst, snd = [], []
    for u, its in interactions.items():
        for it in its:
            for other in item_2_users[it]:
                if other != u:
                    fst.append(u)
                    snd.append(other)
    counts = sp.csr_matrix((np.ones(len(fst)), (fst, snd)), shape=(3, 3)).toarray()
    expected = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=float)
    assert np.array_equal(counts, expected), counts
    kept = GraphDataset._filter_matrix_by_top_k(sp.csr_matrix(counts), 1).toarray()
    # top-1 per row keeps the strongest co-action neighbour only
    assert kept[0].tolist() == [0, 0, 2] and kept[1].tolist() == [0, 0, 2]
    assert kept[2].sum() == 2 and (kept[2] > 0).sum() == 1  # tie between u0 and u1, one survives


def test_lightgcn_propagation_matches_dense_power():
    torch.manual_seed(0)
    adj = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=float)
    norm = _dense_sym_norm(adj)
    graph = GraphDataset._convert_sp_mat_to_sp_tensor(sp.csr_matrix(norm)).coalesce()
    emb = torch.randn(4, 3)

    class Stub:  # only what _apply_graph_encoder touches
        training = True
        _graph_dropout = 0.0
        _num_graph_layers = 2

    from irec.models.mclsr import MCLSRModel

    out = MCLSRModel._apply_graph_encoder(Stub(), emb, graph)
    expected = torch.tensor(norm @ norm @ emb.numpy(), dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-5), 'two propagation steps must equal A_hat^2 E'
    out_mean = MCLSRModel._apply_graph_encoder(Stub(), emb, graph, use_mean=True)
    expected_mean = (emb + torch.tensor(norm @ emb.numpy(), dtype=torch.float32) + expected) / 3
    assert torch.allclose(out_mean, expected_mean, atol=1e-5), 'layer mean must average layers 0..L'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
