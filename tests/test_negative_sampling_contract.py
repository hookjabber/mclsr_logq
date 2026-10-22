"""Tiny CPU contracts for finite negative sampling and identity masks, without training."""

import pickle
import random

import numpy as np
import pytest
import torch

from irec.dataset.negative_samplers import PopularNegativeSampler, RandomNegativeSampler
from irec.dataset.samplers.mclsr import MCLSRTrainSampler
from irec.loss.contrastive import FpsLogQLoss, FpsLoss
from irec.loss.retrieval import FullSoftmaxLoss, InBatchLogQLoss, SampledLogQLoss, SampledSoftmaxLoss


SAMPLERS = ('random', 'popular', 'fallback')


@pytest.fixture(autouse=True)
def single_thread_cpu():
    torch.set_num_threads(1)


def _row(user, items):
    return {'user.ids': [user], 'user.length': 1, 'item.ids': list(items), 'item.length': len(items)}


def _request(kind, dataset, sample, count, catalogue=4, seen=None):
    if kind == 'fallback':
        if seen is None:
            seen = {}
            for row in dataset:
                seen.setdefault(row['user.ids'][0], set()).update(row['item.ids'])
        sampler = MCLSRTrainSampler([sample], 2, catalogue, seen, count)
        return sampler[0]['negatives.ids']
    sampler_type = RandomNegativeSampler if kind == 'random' else PopularNegativeSampler
    return sampler_type(dataset, 2, catalogue).generate_negative_samples(sample, count)


def _rejection_watchdog(monkeypatch, repeated=1):
    calls = []
    original_choice = np.random.choice

    def draw(*arguments, **options):
        calls.append(1)
        if len(calls) > 64:
            raise RuntimeError('rejection budget exceeded')
        return repeated

    def choice(population, size=None, replace=True, p=None):
        if not replace:
            return original_choice(population, size=size, replace=False, p=p)
        draw()
        return np.full(size, repeated, dtype=int)

    monkeypatch.setattr(random, 'randint', draw)
    monkeypatch.setattr(np.random, 'randint', draw)
    monkeypatch.setattr(np.random, 'choice', choice)
    return calls


@pytest.mark.parametrize('kind', SAMPLERS)
@pytest.mark.parametrize('seen_items,requested', [([1, 2, 3, 4], 1), ([1, 2, 3], 2)])
def test_impossible_unique_negative_request_fails_before_rng(kind, seen_items, requested, monkeypatch):
    sample = _row(1, seen_items)
    calls = _rejection_watchdog(monkeypatch)
    with pytest.raises(ValueError, match='negative|available|support'):
        _request(kind, [sample, _row(2, [4])], sample, requested)
    assert not calls


@pytest.mark.parametrize('kind', SAMPLERS)
@pytest.mark.parametrize('count', [-1, 0.5, True])
def test_invalid_requested_count_is_rejected(kind, count, monkeypatch):
    sample = _row(1, [1, 2])
    _rejection_watchdog(monkeypatch, repeated=3)
    with pytest.raises(ValueError, match='negative|integer'):
        _request(kind, [sample, _row(2, [3, 4])], sample, count)


@pytest.mark.parametrize('kind', SAMPLERS)
def test_zero_negatives_is_valid_even_for_saturated_catalogue(kind):
    sample = _row(1, [1, 1, 2])
    assert _request(kind, [sample], sample, 0, catalogue=2) == []


@pytest.mark.parametrize('kind', SAMPLERS)
def test_bounded_rejection_falls_back_without_false_or_duplicate_negatives(kind, monkeypatch):
    sample = _row(1, [1, 1, 2])
    calls = _rejection_watchdog(monkeypatch)
    negatives = _request(kind, [sample, _row(1, [2]), _row(2, [3, 4, 4])], sample, 2)
    assert set(negatives) == {3, 4}
    assert len(negatives) == 2
    assert len(calls) <= 64


@pytest.mark.parametrize('kind', SAMPLERS)
def test_current_sample_is_excluded_even_when_not_in_seen_index(kind, monkeypatch):
    sample = _row(1, [1, 2])
    _rejection_watchdog(monkeypatch, repeated=2)
    negatives = _request(kind, [_row(1, [1]), _row(2, [2, 3])], sample, 1, catalogue=3, seen={1: {1}})
    assert negatives == [3]


def test_popular_cannot_invent_mass_for_catalogue_items_outside_training_support(monkeypatch):
    sample = _row(1, [1, 2])
    calls = _rejection_watchdog(monkeypatch)
    with pytest.raises(ValueError, match='negative|available|support'):
        _request('popular', [sample], sample, 1, catalogue=4)
    assert not calls


@pytest.mark.parametrize('sampler_type', [RandomNegativeSampler, PopularNegativeSampler])
@pytest.mark.parametrize('item_id', [0, 5])
def test_padding_or_out_of_catalogue_ids_cannot_enter_sampling_population(sampler_type, item_id):
    with pytest.raises(ValueError, match='item|catalogue|padding'):
        sampler_type([_row(1, [1, item_id])], 1, 4)


def test_popular_fallback_preserves_conditional_frequency_weights(monkeypatch):
    sample = _row(1, [1, 2])
    sampler = PopularNegativeSampler([sample, _row(2, [3, 4, 4, 4])], 2, 5)
    calls = []

    def choice(population, size=None, replace=True, p=None):
        calls.append(replace)
        if len(calls) > 65:
            raise RuntimeError('rejection budget exceeded')
        if replace:
            return np.full(size, 1, dtype=int)
        assert dict(zip(population.tolist(), p.tolist())) == pytest.approx({3: 0.25, 4: 0.75})
        return np.array([4])

    monkeypatch.setattr(np.random, 'choice', choice)
    assert sampler.generate_negative_samples(sample, 1) == [4]
    assert calls[-1] is False


def _counts(tmp_path):
    path = tmp_path / 'counts.pkl'
    path.write_bytes(pickle.dumps([0, 2, 3, 1, 0]))
    return str(path)


def _assert_zero_loss_and_gradients(value, parameters):
    assert torch.isfinite(value)
    assert value.item() == pytest.approx(0.0, abs=1e-12)
    for gradient in torch.autograd.grad(value, parameters):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize('variant', ['negatives_only', 'standard', 'corrected'])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('mixed', [0, 2])
def test_no_valid_inbatch_negatives_stays_zero_below_finite_mask_sentinel(
    tmp_path, monkeypatch, variant, batch_size, mixed
):
    queries = torch.full((batch_size, 1), 1e6, dtype=torch.float64, requires_grad=True)
    positives = torch.full((batch_size, 1), -2e6, dtype=torch.float64, requires_grad=True)
    table = torch.full((5, 1), -2e6, dtype=torch.float64)
    monkeypatch.setattr(torch, 'randint', lambda low, high, shape, **options: torch.ones(shape, dtype=torch.long))
    loss = InBatchLogQLoss(
        'queries',
        'positives',
        'ids',
        _counts(tmp_path),
        variant=variant,
        user_ids_prefix='users',
        mixed_uniform_negatives=mixed,
        table_prefix='table',
    )
    value = loss(
        {
            'queries': queries,
            'positives': positives,
            'ids': torch.ones(batch_size, dtype=torch.long),
            'users': torch.ones(batch_size, dtype=torch.long),
            'table': table,
        }
    )
    _assert_zero_loss_and_gradients(value, (queries, positives))


def test_explicit_duplicate_positive_negatives_are_excluded_at_any_finite_score(tmp_path):
    queries = torch.tensor([[1e6]], dtype=torch.float64, requires_grad=True)
    positives = torch.tensor([[-2e6]], dtype=torch.float64, requires_grad=True)
    negatives = torch.tensor([[[-2e6], [-2e6]]], dtype=torch.float64, requires_grad=True)
    loss = SampledLogQLoss('queries', 'positives', 'negatives', 'ids', 'negative_ids', _counts(tmp_path))
    value = loss(
        {
            'queries': queries,
            'positives': positives,
            'negatives': negatives,
            'ids': torch.tensor([1]),
            'negative_ids': torch.tensor([[1, 1]]),
        }
    )
    _assert_zero_loss_and_gradients(value, (queries, positives, negatives))


def test_single_real_item_catalogue_excludes_reserved_logits_at_any_finite_score():
    queries = torch.tensor([[1e6]], dtype=torch.float64, requires_grad=True)
    table = torch.tensor([[0.0], [-2e6], [0.0]], dtype=torch.float64, requires_grad=True)
    value = FullSoftmaxLoss('queries', 'table', 'ids')({'queries': queries, 'table': table, 'ids': torch.tensor([1])})
    _assert_zero_loss_and_gradients(value, (queries, table))


def test_corrected_mixed_validity_rows_match_loop_oracle_and_keep_gradients_finite(tmp_path):
    queries = torch.tensor([[0.2, 0.5], [-0.3, 0.8], [0.9, -0.4]], dtype=torch.float64, requires_grad=True)
    positives = torch.tensor([[0.5, -0.6], [0.5, -0.6], [0.1, 0.7]], dtype=torch.float64, requires_grad=True)
    inputs = {
        'queries': queries,
        'positives': positives,
        'ids': torch.tensor([1, 1, 2]),
        'users': torch.tensor([7, 8, 7]),
    }
    loss = InBatchLogQLoss(
        'queries', 'positives', 'ids', _counts(tmp_path), variant='corrected', user_ids_prefix='users'
    )
    actual = loss(inputs)
    terms = []
    for row, column in ((1, 2), (2, 1)):
        positive = (queries[row] * positives[row]).sum()
        proposal = loss._prob_table[inputs['ids'][column]] / (1 - loss._prob_table[inputs['ids'][row]])
        negative = (queries[row] * positives[column]).sum() - proposal.log()
        weight = (1 - (positive - torch.logaddexp(positive, negative)).exp()).detach()
        terms.append(weight * (negative - positive))
    expected = sum(terms) / 3
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    actual_gradients = torch.autograd.grad(actual, (queries, positives), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (queries, positives))
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('kind', ['plain', 'logq'])
def test_explicit_zero_negative_columns_have_zero_finite_gradient(tmp_path, kind):
    queries = torch.tensor([[0.2, -0.8]], requires_grad=True)
    positives = torch.tensor([[0.4, 0.5]], requires_grad=True)
    negatives = torch.empty(1, 0, 2, requires_grad=True)
    inputs = {
        'queries': queries,
        'positives': positives,
        'negatives': negatives,
        'ids': torch.tensor([1]),
        'negative_ids': torch.empty(1, 0, dtype=torch.long),
    }
    loss = (
        SampledSoftmaxLoss('queries', 'positives', 'negatives')
        if kind == 'plain'
        else SampledLogQLoss('queries', 'positives', 'negatives', 'ids', 'negative_ids', _counts(tmp_path))
    )
    _assert_zero_loss_and_gradients(loss(inputs), (queries, positives, negatives))


@pytest.mark.parametrize('scheme', ['paper', 'cross_only', 'symmetric'])
@pytest.mark.parametrize('batch_size', [1, 3])
def test_masked_cosine_auxiliary_with_no_distinct_ids_is_finite_zero(tmp_path, scheme, batch_size):
    first = torch.tensor([[0.3, -0.7]] * batch_size, dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[-0.4, 0.2]] * batch_size, dtype=torch.float64, requires_grad=True)
    loss = FpsLogQLoss(
        'first',
        'second',
        'ids',
        _counts(tmp_path),
        tau=0.5,
        normalize_embeddings=True,
        scheme=scheme,
        mask_false_negatives=True,
    )
    value = loss({'first': first, 'second': second, 'ids': torch.ones(batch_size, dtype=torch.long)})
    _assert_zero_loss_and_gradients(value, (first, second))


@pytest.mark.parametrize('scheme', ['paper', 'cross_only', 'symmetric'])
def test_unmasked_auxiliary_is_a_row_objective_not_silent_entity_deduplication(scheme):
    embeddings = torch.tensor([[0.3, 0.7]] * 3, dtype=torch.float64)
    value = FpsLoss('first', 'second', tau=0.5, normalize_embeddings=True, scheme=scheme)(
        {'first': embeddings, 'second': embeddings, 'ids': torch.ones(3, dtype=torch.long)}
    )
    candidate_count = 3 if scheme == 'cross_only' else 5
    expected = np.log(candidate_count) / (2 if scheme == 'symmetric' else 1)
    assert value.item() == pytest.approx(expected)
