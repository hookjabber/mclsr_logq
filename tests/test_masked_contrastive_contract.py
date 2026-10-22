import math
import pickle

import pytest
import torch

from irec.loss.contrastive import (
    ContrastiveFullSoftmaxLoss,
    FpsLogQLoss,
    FpsLoss,
    MatchedContrastiveFullSoftmaxLoss,
)


SCHEMES = ('cross_only', 'paper', 'symmetric')


def _artifact(tmp_path, name, values):
    path = tmp_path / name
    path.write_bytes(pickle.dumps(values))
    return str(path)


def _assert_zero_value_and_gradients(value, parameters):
    assert torch.isfinite(value)
    assert value.item() == 0.0
    for gradient in torch.autograd.grad(value, parameters):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


def _assert_value_and_gradients(actual, expected, parameters):
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-10)
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, atol=2e-3, rtol=2e-6)


def _reduce(terms, use_mean, symmetric=False):
    stacked = torch.stack(terms)
    reduced = stacked.mean() if use_mean else stacked.sum()
    return reduced / 2 if symmetric else reduced


def _row_objective(candidates, positive_index=0):
    scores = torch.stack(candidates)
    shifted = scores - scores.detach().max()
    return torch.logsumexp(shifted, dim=0) - shifted[positive_index]


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('similarity', ('dot', 'euclidean'))
@pytest.mark.parametrize('batch_size', (1, 3))
@pytest.mark.parametrize('use_mean', (False, True))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_logq_no_distinct_ids_has_exact_zero_value_and_gradients(
    tmp_path, scheme, similarity, batch_size, use_mean, dtype
):
    first = torch.full((batch_size, 1), 1e6, dtype=dtype, requires_grad=True)
    second = torch.full((batch_size, 1), -2e6, dtype=dtype, requires_grad=True)
    loss = FpsLogQLoss(
        'first',
        'second',
        'ids',
        _artifact(tmp_path, 'counts.pkl', [0, 1, 0]),
        tau=0.5,
        scheme=scheme,
        similarity=similarity,
        use_mean=use_mean,
        mask_false_negatives=True,
        leave_own_out=True,
        center_log_q=True,
    )
    value = loss({'first': first, 'second': second, 'ids': torch.ones(batch_size, dtype=torch.long)})
    _assert_zero_value_and_gradients(value, (first, second))


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('mode', ('sample', 'inbatch_negative'))
@pytest.mark.parametrize('leave_own_out', (False, True))
def test_original_dot_repro_without_centering_has_zero_gradients(tmp_path, scheme, mode, leave_own_out):
    first = torch.full((2, 1), 1e6, dtype=torch.float64, requires_grad=True)
    second = torch.full((2, 1), -2e6, dtype=torch.float64, requires_grad=True)
    loss = FpsLogQLoss(
        'first',
        'second',
        'ids',
        _artifact(tmp_path, 'counts.pkl', [0, 1, 0]),
        tau=0.5,
        scheme=scheme,
        logq_probability_mode=mode,
        leave_own_out=leave_own_out,
    )
    value = loss({'first': first, 'second': second, 'ids': torch.ones(2, dtype=torch.long)})
    _assert_zero_value_and_gradients(value, (first, second))


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('similarity', ('dot', 'euclidean'))
@pytest.mark.parametrize('use_mean', (False, True))
def test_plain_singleton_keeps_only_the_paired_positive(scheme, similarity, use_mean):
    first = torch.tensor([[1e6]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-2e6]], dtype=torch.float64, requires_grad=True)
    loss = FpsLoss('first', 'second', tau=0.5, scheme=scheme, similarity=similarity, use_mean=use_mean)
    _assert_zero_value_and_gradients(loss({'first': first, 'second': second}), (first, second))


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('similarity', ('dot', 'euclidean'))
@pytest.mark.parametrize('use_mean', (False, True))
def test_partial_identity_masks_match_physically_enumerated_value_and_gradients(tmp_path, scheme, similarity, use_mean):
    first = torch.tensor([[1e6], [1e6 + 1], [-1e6]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-2e6], [-2e6], [-2e6]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([1, 1, 2])
    counts = [0, 1, 3, 0]
    loss = FpsLogQLoss(
        'first',
        'second',
        'ids',
        _artifact(tmp_path, 'counts.pkl', counts),
        tau=0.5,
        scheme=scheme,
        similarity=similarity,
        use_mean=use_mean,
        logq_probability_mode='sample',
        logq_lambda=0.6,
    )

    def score(anchor, candidate):
        product = (anchor * candidate).sum() if similarity == 'dot' else -(anchor - candidate).square().sum()
        return product / 0.5

    views = (first, second)
    terms = []
    anchor_views = (0, 1) if scheme == 'symmetric' else (0,)
    for anchor_view in anchor_views:
        for anchor_row, anchor_id in enumerate(ids.tolist()):
            anchor = views[anchor_view][anchor_row]
            positive_score = score(anchor, views[1 - anchor_view][anchor_row])
            candidates = [positive_score]
            candidate_views = (1 - anchor_view,) if scheme == 'cross_only' else (0, 1)
            for candidate_view in candidate_views:
                for candidate_row, candidate_id in enumerate(ids.tolist()):
                    if candidate_id == anchor_id:
                        continue
                    correction = 0.6 * math.log(counts[candidate_id] / sum(counts))
                    candidates.append(score(anchor, views[candidate_view][candidate_row]) - correction)
            terms.append(_row_objective(candidates))
    expected = _reduce(terms, use_mean, scheme == 'symmetric')
    actual = loss({'first': first, 'second': second, 'ids': ids})
    _assert_value_and_gradients(actual, expected, (first, second))


@pytest.mark.parametrize('batch_size', (1, 3))
@pytest.mark.parametrize('use_mean', (False, True))
def test_full_catalogue_single_entity_has_zero_value_and_gradients(batch_size, use_mean):
    anchors = torch.full((batch_size, 1), 1e6, dtype=torch.float64, requires_grad=True)
    table = torch.tensor([[3e6], [-2e6], [4e6]], dtype=torch.float64, requires_grad=True)
    loss = ContrastiveFullSoftmaxLoss('anchors', 'table', 'ids', tau=0.5, use_mean=use_mean)
    value = loss({'anchors': anchors, 'table': table, 'ids': torch.ones(batch_size, dtype=torch.long)})
    _assert_zero_value_and_gradients(value, (anchors, table))


@pytest.mark.parametrize('batch_size', (1, 3))
@pytest.mark.parametrize('use_mean', (False, True))
def test_matched_single_present_entity_has_zero_value_and_gradients(tmp_path, batch_size, use_mean):
    first = torch.full((batch_size, 1), 1e6, dtype=torch.float64, requires_grad=True)
    second = torch.full((batch_size, 1), -2e6, dtype=torch.float64, requires_grad=True)
    first_table = torch.tensor([[3e6], [1e6], [7e6], [4e6]], dtype=torch.float64, requires_grad=True)
    second_table = torch.tensor([[3e6], [-2e6], [7e6], [4e6]], dtype=torch.float64, requires_grad=True)
    loss = MatchedContrastiveFullSoftmaxLoss(
        'first',
        'second',
        'first_table',
        'second_table',
        'ids',
        _artifact(tmp_path, 'presence.pkl', [False, True, False, False]),
        tau=0.5,
        use_mean=use_mean,
    )
    value = loss(
        {
            'first': first,
            'second': second,
            'first_table': first_table,
            'second_table': second_table,
            'ids': torch.ones(batch_size, dtype=torch.long),
        }
    )
    _assert_zero_value_and_gradients(value, (first, second, first_table, second_table))


@pytest.mark.parametrize('matched', (False, True))
@pytest.mark.parametrize('use_mean', (False, True))
def test_full_catalogue_partial_masks_match_physically_enumerated_oracle(tmp_path, matched, use_mean):
    first = torch.tensor([[1e6, 0.3], [1e6, -0.7]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-2e6, 0.4], [-2e6, -0.2]], dtype=torch.float64, requires_grad=True)
    first_table = torch.tensor([[3e6, 0], [1e6, 0.3], [-1e6, 0.8], [4e6, 0]], dtype=torch.float64, requires_grad=True)
    second_table = torch.tensor(
        [[3e6, 0], [-2e6, 0.4], [-2e6, -0.8], [4e6, 0]], dtype=torch.float64, requires_grad=True
    )
    ids = torch.tensor([1, 1])
    if matched:
        loss = MatchedContrastiveFullSoftmaxLoss(
            'first',
            'second',
            'first_table',
            'second_table',
            'ids',
            _artifact(tmp_path, 'presence.pkl', [False, True, True, False]),
            tau=0.5,
            use_mean=use_mean,
        )
        directions = ((first, second_table, first_table), (second, first_table, second_table))
        parameters = (first, second, first_table, second_table)
    else:
        loss = ContrastiveFullSoftmaxLoss('first', 'second_table', 'ids', tau=0.5, use_mean=use_mean)
        directions = ((first, second_table, None),)
        parameters = (first, second_table)
    terms = []
    for anchors, other_table, same_table in directions:
        for anchor_row, anchor_id in enumerate(ids.tolist()):
            anchor = anchors[anchor_row]
            candidates = [(anchor * other_table[entity_id]).sum() / 0.5 for entity_id in (1, 2)]
            if same_table is not None:
                candidates.extend(
                    (anchor * same_table[entity_id]).sum() / 0.5 for entity_id in (1, 2) if entity_id != anchor_id
                )
            terms.append(_row_objective(candidates, positive_index=anchor_id - 1))
    expected = _reduce(terms, use_mean, matched)
    actual = loss(
        {'first': first, 'second': second, 'first_table': first_table, 'second_table': second_table, 'ids': ids}
    )
    _assert_value_and_gradients(actual, expected, parameters)


@pytest.mark.parametrize('table_size, positive_id', ((2, 0), (2, 1), (3, 0), (3, 2)))
def test_full_catalogue_masked_positive_is_rejected_before_softmax(table_size, positive_id):
    loss = ContrastiveFullSoftmaxLoss('anchors', 'table', 'ids')
    with pytest.raises(ValueError, match='positive'):
        loss(
            {
                'anchors': torch.ones(1, 1, requires_grad=True),
                'table': torch.ones(table_size, 1, requires_grad=True),
                'ids': torch.tensor([positive_id]),
            }
        )


def test_matched_all_absent_catalogue_rejects_masked_positive(tmp_path):
    loss = MatchedContrastiveFullSoftmaxLoss(
        'first',
        'second',
        'first_table',
        'second_table',
        'ids',
        _artifact(tmp_path, 'presence.pkl', [False, False, False]),
    )
    with pytest.raises(ValueError, match='positives would be masked'):
        loss(
            {
                'first': torch.ones(1, 1, requires_grad=True),
                'second': torch.ones(1, 1, requires_grad=True),
                'first_table': torch.ones(3, 1, requires_grad=True),
                'second_table': torch.ones(3, 1, requires_grad=True),
                'ids': torch.tensor([1]),
            }
        )
