import math
import pickle

import numpy as np
import pytest
import torch

from irec.loss import InBatchLogQLoss
from irec.loss.counts import load_counts_artifact


def _loss(tmp_path, counts=None, **options):
    counts_path = tmp_path / 'counts.pkl'
    with counts_path.open('wb') as destination:
        pickle.dump([0, 1, 2, 5, 0] if counts is None else counts, destination)
    return InBatchLogQLoss('queries', 'positives', 'item_ids', str(counts_path), **options)


def _analytical_value_and_gradients(
    queries, positives, item_ids, user_ids, variant, coefficient, temperature, normalize
):
    probabilities = {1: 1 / 8, 2: 2 / 8, 3: 5 / 8}
    batch_size = len(item_ids)
    dimension = len(queries[0])

    def normalized(vectors):
        return [[value / math.sqrt(sum(entry**2 for entry in vector)) for value in vector] for vector in vectors]

    query_vectors = normalized(queries) if normalize else queries
    positive_vectors = normalized(positives) if normalize else positives
    score_gradients = [[0.0] * batch_size for _ in range(batch_size)]
    total = 0.0
    for row in range(batch_size):
        scores = [
            sum(left * right for left, right in zip(query_vectors[row], candidate)) / temperature
            for candidate in positive_vectors
        ]
        valid = [
            column
            for column in range(batch_size)
            if column != row and item_ids[column] != item_ids[row] and user_ids[column] != user_ids[row]
        ]
        negative_masses = {}
        for column in valid:
            proposal = probabilities[item_ids[column]]
            if variant in ('leave_own_out', 'corrected'):
                proposal /= 1 - probabilities[item_ids[row]]
            negative_masses[column] = math.exp(scores[column]) / proposal**coefficient
        positive_mass = math.exp(scores[row])
        if variant == 'standard':
            positive_mass /= probabilities[item_ids[row]] ** coefficient
        negative_sum = sum(negative_masses.values())
        if variant == 'corrected':
            weight = negative_sum / (len(valid) * positive_mass + negative_sum)
            total += weight * (math.log(negative_sum) - scores[row])
            score_gradients[row][row] = -weight / batch_size
            for column in valid:
                score_gradients[row][column] = weight * negative_masses[column] / negative_sum / batch_size
        else:
            denominator = positive_mass + negative_sum
            total += math.log(denominator / positive_mass)
            score_gradients[row][row] = (positive_mass / denominator - 1) / batch_size
            for column in valid:
                score_gradients[row][column] = negative_masses[column] / denominator / batch_size
    query_gradients = [
        [
            sum(score_gradients[row][column] * positive_vectors[column][axis] for column in range(batch_size))
            / temperature
            for axis in range(dimension)
        ]
        for row in range(batch_size)
    ]
    positive_gradients = [
        [
            sum(score_gradients[row][column] * query_vectors[row][axis] for row in range(batch_size)) / temperature
            for axis in range(dimension)
        ]
        for column in range(batch_size)
    ]
    if normalize:
        for vectors, gradients in ((queries, query_gradients), (positives, positive_gradients)):
            for vector, gradient in zip(vectors, gradients):
                norm = math.sqrt(sum(value**2 for value in vector))
                projection = sum(value * derivative for value, derivative in zip(vector, gradient))
                gradient[:] = [
                    derivative / norm - value * projection / norm**3 for value, derivative in zip(vector, gradient)
                ]
    return tuple(
        torch.tensor(result, dtype=torch.float64)
        for result in (total / batch_size, query_gradients, positive_gradients)
    )


@pytest.mark.parametrize('variant', ['negatives_only', 'standard', 'leave_own_out', 'corrected'])
@pytest.mark.parametrize('coefficient', [0.0, 0.35, 1.0])
@pytest.mark.parametrize('temperature', [0.5, 2.0])
@pytest.mark.parametrize('normalize', [False, True])
def test_loss_and_both_embedding_gradients_match_analytical_oracle(
    tmp_path, variant, coefficient, temperature, normalize
):
    query_values = [[0.3, -0.2], [-0.7, 0.4], [0.2, 0.6], [0.5, 0.1]]
    positive_values = [[0.1, 0.6], [0.8, -0.3], [0.8, -0.3], [-0.5, 0.2]]
    item_ids = [1, 2, 2, 3]
    user_ids = [7, 8, 9, 9]
    queries = torch.tensor(query_values, dtype=torch.float64, requires_grad=True)
    positives = torch.tensor(positive_values, dtype=torch.float64, requires_grad=True)
    loss = _loss(
        tmp_path,
        variant='negatives_only' if variant == 'leave_own_out' else variant,
        leave_own_out=variant == 'leave_own_out',
        logq_lambda=coefficient,
        temperature=temperature,
        normalize_embeddings=normalize,
        user_ids_prefix='user_ids',
    )
    actual = loss(
        {
            'queries': queries,
            'positives': positives,
            'item_ids': torch.tensor(item_ids),
            'user_ids': torch.tensor(user_ids),
        }
    )
    actual_gradients = torch.autograd.grad(actual, (queries, positives))
    expected = _analytical_value_and_gradients(
        query_values, positive_values, item_ids, user_ids, variant, coefficient, temperature, normalize
    )
    for observed, wanted in zip((actual, *actual_gradients), expected):
        torch.testing.assert_close(observed, wanted, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('variant', ['negatives_only', 'standard', 'corrected'])
@pytest.mark.parametrize('item_ids,user_ids', [([1], [7]), ([1, 1], [7, 8]), ([1, 2], [7, 7])])
def test_no_valid_negatives_gives_zero_loss_and_gradients(tmp_path, variant, item_ids, user_ids):
    queries = torch.zeros(len(item_ids), 2, requires_grad=True)
    positives = torch.ones(len(item_ids), 2, requires_grad=True)
    loss = _loss(tmp_path, variant=variant, user_ids_prefix='user_ids')
    actual = loss(
        {
            'queries': queries,
            'positives': positives,
            'item_ids': torch.tensor(item_ids),
            'user_ids': torch.tensor(user_ids),
        }
    )
    actual.backward()
    torch.testing.assert_close(actual, torch.zeros(()))
    torch.testing.assert_close(queries.grad, torch.zeros_like(queries))
    torch.testing.assert_close(positives.grad, torch.zeros_like(positives))


def test_corrected_lambda_zero_is_not_the_uncorrected_softmax_baseline(tmp_path):
    inputs = {
        'queries': torch.zeros(3, 3, requires_grad=True),
        'positives': torch.eye(3),
        'item_ids': torch.tensor([1, 2, 3]),
    }
    baseline = _loss(tmp_path, logq_lambda=0.0)(inputs)
    corrected = _loss(tmp_path, logq_lambda=0.0, variant='corrected')(inputs)
    baseline_gradient = torch.autograd.grad(baseline, inputs['queries'])[0]
    corrected_gradient = torch.autograd.grad(corrected, inputs['queries'])[0]
    assert baseline.item() == pytest.approx(math.log(3))
    assert corrected.item() == pytest.approx(0.5 * math.log(2))
    torch.testing.assert_close(baseline_gradient.diagonal(), torch.full((3,), -2 / 9))
    torch.testing.assert_close(corrected_gradient.diagonal(), torch.full((3,), -1 / 6))


@pytest.mark.parametrize('variant,expected', [('standard', -2 / 9), ('negatives_only', -2 / 7)])
def test_uniform_proposal_still_changes_negatives_only_gradient(tmp_path, variant, expected):
    inputs = {
        'queries': torch.zeros(3, 3, requires_grad=True),
        'positives': torch.eye(3),
        'item_ids': torch.tensor([1, 2, 3]),
    }
    actual = _loss(tmp_path, counts=[0, 1, 1, 1, 0], variant=variant)(inputs)
    actual.backward()
    torch.testing.assert_close(inputs['queries'].grad.diagonal(), torch.full((3,), expected))


def test_zero_support_floor_does_not_change_supported_probabilities(tmp_path):
    loss = _loss(tmp_path, counts=[0, 1, 3, 0, 0])
    torch.testing.assert_close(loss._prob_table[1:3], torch.tensor([0.25, 0.75]), rtol=0, atol=0)
    torch.testing.assert_close(loss._prob_table[[0, 3, 4]], torch.full((3,), 1e-10), rtol=0, atol=0)
    assert torch.isfinite(loss._log_q_table).all()


@pytest.mark.parametrize(
    'expectations',
    [{'expected_counts_role': 'context-inclusion'}, {'expected_counts_max_len': 20}],
)
def test_downstream_enforces_count_metadata(tmp_path, expectations):
    artifact = {'counts': [0, 1, 2, 5, 0], 'role': 'target', 'max_len': 21, 'denominator': 8}
    with pytest.raises(ValueError, match='counts artifact'):
        _loss(tmp_path, counts=artifact, **expectations)


@pytest.mark.parametrize(
    'counts',
    [[0, 0, 0, 0], [0, 1, float('nan'), 0], [0, 1, float('inf'), 0], [0, -1, 2, 0]],
    ids=['zero-total', 'nan', 'infinity', 'negative'],
)
def test_invalid_count_distributions_are_rejected(tmp_path, counts):
    with pytest.raises(ValueError, match='counts'):
        _loss(tmp_path, counts=counts)


def _load_counts(tmp_path, artifact):
    path = tmp_path / 'counts.pkl'
    with path.open('wb') as destination:
        pickle.dump(artifact, destination)
    return load_counts_artifact(path)


@pytest.mark.parametrize(
    'counts',
    [
        [],
        1,
        [[1, 2]],
        [[1], [2, 3]],
        ['1', '2'],
        [True, False],
        [1 + 0j, 2 + 0j],
        np.array([1, 2], dtype=object),
        [1e308, 1e308],
    ],
    ids=['empty', 'scalar', 'matrix', 'ragged', 'text', 'boolean', 'complex', 'object', 'overflow-total'],
)
def test_invalid_count_shape_or_numeric_type_is_rejected(tmp_path, counts):
    with pytest.raises(ValueError, match='counts'):
        _load_counts(tmp_path, counts)


@pytest.mark.parametrize(
    'denominator',
    [0, -1, float('nan'), float('inf'), None, '8', True, 8 + 0j, [8], 4.99],
    ids=['zero', 'negative', 'nan', 'infinity', 'none', 'text', 'boolean', 'complex', 'vector', 'less-than-count'],
)
def test_invalid_count_denominator_is_rejected(tmp_path, denominator):
    with pytest.raises(ValueError, match='denominator'):
        _load_counts(tmp_path, {'counts': [0, 1, 2, 5, 0], 'denominator': denominator})


@pytest.mark.parametrize('denominator', [5, 8.0, np.float32(5), np.int64(20)])
def test_count_denominator_can_be_smaller_than_count_sum(tmp_path, denominator):
    artifact = {'counts': [0, 1, 2, 5, 0], 'denominator': denominator, 'role': 'context-inclusion'}
    counts, metadata = _load_counts(tmp_path, artifact)
    assert counts == artifact['counts']
    assert metadata == {'denominator': denominator, 'role': 'context-inclusion'}
    assert type(metadata['denominator']) is type(denominator)


@pytest.mark.parametrize(
    'counts',
    [
        [0, 1, 3, 0],
        np.array([0, 1, 3, 0], dtype=np.float32),
        np.array([0, 1, 3, 0], dtype=np.float64),
        np.array([0, 1, 3, 0], dtype=np.int64),
        np.array([2**63, 2**63], dtype=np.uint64),
    ],
    ids=['list', 'float32', 'float64', 'int64', 'uint64-large-total'],
)
@pytest.mark.parametrize('with_metadata', [False, True])
def test_valid_counts_preserve_stored_type_dtype_and_metadata(tmp_path, counts, with_metadata):
    metadata = {'role': 'target', 'max_len': 21, 'source': 'train_mclsr.txt'} if with_metadata else {}
    restored, restored_metadata = _load_counts(tmp_path, {'counts': counts, **metadata} if with_metadata else counts)
    assert type(restored) is type(counts)
    if isinstance(counts, np.ndarray):
        assert restored.dtype == counts.dtype
    np.testing.assert_array_equal(restored, counts)
    assert restored_metadata == metadata


def test_zero_total_is_rejected_even_when_logq_is_disabled(tmp_path):
    with pytest.raises(ValueError, match='counts'):
        _loss(tmp_path, counts=[0, 0, 0, 0], logq_lambda=0.0)
