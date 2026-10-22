import pickle

import numpy as np
import pytest
import torch

from irec.loss.contrastive import (
    ContrastiveFullSoftmaxLoss,
    FpsLogQLoss,
    FpsLoss,
    MatchedContrastiveFullSoftmaxLoss,
)
from irec.loss.retrieval import InBatchLogQLoss, SampledLogQLoss


def _counts_file(tmp_path, counts):
    destination = tmp_path / 'counts.pkl'
    with destination.open('wb') as output:
        pickle.dump(counts, output)
    return str(destination)


def _corrected_loss(kind, counts_path, **options):
    if kind == 'inbatch':
        return InBatchLogQLoss('queries', 'positives', 'ids', counts_path, **options)
    if kind == 'sampled':
        return SampledLogQLoss('queries', 'positives', 'negatives', 'ids', 'negative_ids', counts_path, **options)
    return FpsLogQLoss('queries', 'positives', 'ids', counts_path, tau=0.5, scheme='paper', **options)


@pytest.mark.parametrize('kind', ['inbatch', 'sampled', 'contrastive'])
@pytest.mark.parametrize('scale', [1e39, 1e-100])
def test_count_scale_does_not_change_probability_loss_or_gradients(tmp_path, kind, scale):
    counts = np.array([0, 1, 3, 0], dtype=np.float64)
    baseline = _corrected_loss(kind, _counts_file(tmp_path, counts))
    scaled = _corrected_loss(kind, _counts_file(tmp_path, counts * scale))
    assert torch.isfinite(scaled._log_q_table).all()
    torch.testing.assert_close(scaled._log_q_table, baseline._log_q_table)

    def evaluate(loss):
        queries = torch.tensor([[0.4, -0.8], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)
        positives = torch.tensor([[0.3, 0.9], [-0.6, 0.5]], dtype=torch.float64, requires_grad=True)
        inputs = {
            'queries': queries,
            'positives': positives,
            'ids': torch.tensor([1, 2]),
            'negatives': positives.flip(0).unsqueeze(1),
            'negative_ids': torch.tensor([[2], [1]]),
        }
        value = loss(inputs)
        return value.detach(), torch.autograd.grad(value, (queries, positives))

    reference_value, reference_gradients = evaluate(baseline)
    actual_value, actual_gradients = evaluate(scaled)
    torch.testing.assert_close(actual_value, reference_value)
    for actual, expected in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('denominator', [0, -1, float('nan'), float('inf'), True, '4', [4], 2])
def test_explicit_counts_denominator_must_be_valid_probability_denominator(tmp_path, denominator):
    counts_path = _counts_file(tmp_path, [0, 1, 3, 0])
    with pytest.raises(ValueError, match='denominator'):
        _corrected_loss('contrastive', counts_path, counts_denominator=denominator)


@pytest.mark.parametrize('kind', ['inbatch', 'contrastive', 'fps', 'full', 'matched_full'])
@pytest.mark.parametrize('temperature', [0, -0.5, float('nan'), float('inf')])
def test_temperature_must_be_finite_and_positive(tmp_path, kind, temperature):
    counts_path = _counts_file(tmp_path, [0, 1, 3, 0])
    with pytest.raises(ValueError, match='temperature'):
        if kind == 'inbatch':
            InBatchLogQLoss('queries', 'positives', 'ids', counts_path, temperature=temperature)
        elif kind == 'contrastive':
            FpsLogQLoss('queries', 'positives', 'ids', counts_path, tau=temperature)
        elif kind == 'full':
            ContrastiveFullSoftmaxLoss('queries', 'table', 'ids', tau=temperature)
        elif kind == 'matched_full':
            MatchedContrastiveFullSoftmaxLoss(
                'queries', 'positives', 'table', 'other_table', 'ids', counts_path, tau=temperature
            )
        else:
            FpsLoss('queries', 'positives', tau=temperature)
