"""Numeric reference tests for the graph branch: adjacency normalisation, similarity-graph
construction and LightGCN propagation are compared with hand-computed dense results."""

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from irec.dataset.base import GraphDataset  # noqa: E402


def _dense_sym_norm(adj):
    d = adj.sum(1)
    with np.errstate(divide='ignore'):
        d_inv = np.where(d > 0, d ** -0.5, 0.0)
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
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
    print('\nALL GRAPH TESTS PASSED')
