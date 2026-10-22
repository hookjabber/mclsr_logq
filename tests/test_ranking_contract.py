"""Independent ranking/set-relevance oracles for repeated events and small catalogues."""

import itertools
import math

import pytest
import torch

import irec.models.base as model_base
import irec.utils
from irec.dataloader.batch_processors import BasicBatchProcessor
from irec.metric.base import ComiRecNDCGMetric, MCLSRHitRateMetric, MCLSRNDCGMetric, MCLSRRecallMetric
from irec.models import BaseModel


METRICS = (MCLSRRecallMetric, MCLSRNDCGMetric, MCLSRHitRateMetric, ComiRecNDCGMetric)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(irec.utils, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(model_base, 'DEVICE', torch.device('cpu'))


def _inputs(predictions, targets):
    return {
        'predictions': torch.tensor(predictions, dtype=torch.long),
        'labels.ids': torch.tensor([item for row in targets for item in row], dtype=torch.long),
        'labels.length': torch.tensor([len(row) for row in targets], dtype=torch.long),
    }


def _oracle(predictions, targets, cutoff):
    relevant = set(targets)
    hits = [item in relevant for item in predictions[:cutoff]]
    weights = [1 / math.log2(rank + 2) for rank in range(cutoff)]
    dcg = math.fsum(weight for hit, weight in zip(hits, weights) if hit)
    ideal = math.fsum(weights[: min(cutoff, len(relevant))])
    comirec_ideal = math.fsum(weights[: sum(hits)])
    return (
        sum(hits) / len(relevant) if relevant else 0.0,
        dcg / ideal if ideal else 0.0,
        float(any(hits)),
        dcg / comirec_ideal if comirec_ideal else 0.0,
    )


@pytest.mark.parametrize('metric_type', METRICS)
def test_perfect_unique_ranking_scores_one_with_repeated_targets(metric_type):
    batch = _inputs([[2, 1, 3]], [[1, 2, 1, 1]])
    assert metric_type(3)(batch, 'predictions', 'labels') == pytest.approx([1.0])


@pytest.mark.parametrize('cutoff', [1, 2, 3])
def test_exhaustive_small_rankings_targets_and_batch_partition(cutoff):
    cases = [
        (ranking, targets)
        for ranking in itertools.permutations((1, 2, 3))
        for length in range(4)
        for targets in itertools.product((1, 2, 4), repeat=length)
    ]
    for metric_index, metric_type in enumerate(METRICS):
        expected = [_oracle(ranking, targets, cutoff)[metric_index] for ranking, targets in cases]
        actual = []
        for offset in range(0, len(cases), 17):
            part = cases[offset : offset + 17]
            batch = _inputs([ranking for ranking, _ in part], [targets for _, targets in part])
            actual.extend(metric_type(cutoff)(batch, 'predictions', 'labels'))
        assert actual == pytest.approx(expected, abs=2e-6)
        assert all(0 <= value <= 1.000001 for value in actual)


@pytest.mark.parametrize('metric_type', METRICS)
@pytest.mark.parametrize(
    'malformation',
    [
        'duplicate_prediction',
        'padding_prediction',
        'padding_target',
        'negative_target',
        'extra_label',
        'short_labels',
        'negative_length',
        'floating_ids',
    ],
)
def test_malformed_metric_inputs_fail_instead_of_silent_scores(metric_type, malformation):
    batch = _inputs([[1, 2, 3]], [[1, 2]])
    if malformation == 'duplicate_prediction':
        batch['predictions'][0, 1] = 1
    elif malformation == 'padding_prediction':
        batch['predictions'][0, 0] = 0
    elif malformation == 'padding_target':
        batch['labels.ids'][0] = 0
    elif malformation == 'negative_target':
        batch['labels.ids'][0] = -1
    elif malformation == 'extra_label':
        batch['labels.length'][0] = 1
    elif malformation == 'short_labels':
        batch['labels.length'][0] = 3
    elif malformation == 'negative_length':
        batch['labels.length'][0] = -1
    else:
        batch['labels.ids'] = batch['labels.ids'].float()
    with pytest.raises(ValueError):
        metric_type(3)(batch, 'predictions', 'labels')


@pytest.mark.parametrize('metric_type', METRICS)
@pytest.mark.parametrize('cutoff', [0, -1, 1.5, True])
def test_invalid_metric_cutoff_fails_early(metric_type, cutoff):
    with pytest.raises(ValueError, match='positive integer'):
        metric_type(cutoff)


def _model(kind, num_items=4, cutoff=4):
    return BaseModel.create_from_config(
        {
            'type': kind,
            'sequence_prefix': 'item',
            'user_prefix': 'user',
            'labels_prefix': 'labels',
            'positive_prefix': 'labels',
            'use_graph': False,
            'embedding_dim': 8,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.0,
            'eval_top_k': cutoff,
        },
        num_items=num_items,
        num_users=3,
        max_sequence_length=5,
    ).eval()


def _batch(histories, targets):
    return BasicBatchProcessor()(
        [
            {
                'user.ids': [index + 1],
                'user.length': 1,
                'item.ids': history,
                'item.length': len(history),
                'labels.ids': labels,
                'labels.length': len(labels),
            }
            for index, (history, labels) in enumerate(zip(histories, targets))
        ]
    )


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'sasrec_in_batch'])
def test_repeated_seen_targets_remain_eligible_and_do_not_leak_into_ranking(kind):
    torch.manual_seed(91)
    model = _model(kind)
    batch = _batch([[1, 2, 1], [3, 3]], [[1, 1], [3, 3, 3]])
    changed_labels = _batch([[1, 2, 1], [3, 3]], [[4], []])
    with torch.no_grad():
        ranked = model(batch)
        torch.testing.assert_close(ranked, model(changed_labels))
    assert [set(row) for row in ranked.tolist()] == [{1, 2, 3, 4}] * 2
    inputs = {**batch, 'predictions': ranked}
    assert MCLSRRecallMetric(4)(inputs, 'predictions', 'labels') == [1.0, 1.0]


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'sasrec_in_batch'])
@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_real_catalogue_scores_fail_explicitly(kind, nonfinite):
    model = _model(kind)
    with torch.no_grad():
        model._item_embeddings.weight[4].fill_(nonfinite)
        with pytest.raises(ValueError, match='finite'):
            model(_batch([[1, 2]], [[1]]))


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'sasrec_in_batch'])
@pytest.mark.parametrize('cutoff', [0, -1, 5, True])
def test_invalid_inference_cutoff_fails_with_clear_contract(kind, cutoff):
    with torch.no_grad(), pytest.raises(ValueError, match='eval_top_k'):
        model = _model(kind, cutoff=cutoff)
        model(_batch([[1, 2]], [[1]]))
