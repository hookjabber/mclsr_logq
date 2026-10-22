import numpy as np
import pytest
import scipy.sparse as sp
import torch

from irec.dataset.graph import GraphDataset


@pytest.mark.parametrize('entity_id', [2**24 + 1, 2**24 + 3])
def test_sparse_conversion_preserves_large_catalogue_ids(entity_id):
    rows = np.array([1, entity_id], dtype=np.int64)
    columns = np.array([entity_id, 2], dtype=np.int64)
    adjacency = sp.coo_matrix(
        (np.array([0.25, 0.75]), (rows, columns)),
        shape=(entity_id + 2, entity_id + 2),
    )

    actual = GraphDataset._convert_sp_mat_to_sp_tensor(adjacency).coalesce()

    np.testing.assert_array_equal(actual.indices().numpy(), np.stack([rows, columns]))
    torch.testing.assert_close(actual.values(), torch.tensor([0.25, 0.75], dtype=torch.float32))
    assert tuple(actual.shape) == adjacency.shape


def test_row_top_k_preserves_asymmetry_and_normalizes_pruned_degrees():
    adjacency = sp.csr_matrix(
        np.array([[0.0, 5.0, 2.0, 0.0], [5.0, 0.0, 3.0, 0.0], [2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    )
    expected_kept = np.array([[0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    kept = GraphDataset._filter_matrix_by_top_k(adjacency, 1)
    normalized = GraphDataset.get_sparse_graph_layer(kept, 4, 4).toarray()

    np.testing.assert_array_equal(kept.toarray(), expected_kept)
    expected_normalized = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, np.sqrt(3.0 / 5.0), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(normalized, expected_normalized)
    assert not np.array_equal(normalized, normalized.T)
