import itertools
import math
import pickle
from collections import Counter

import pytest
import torch

from irec.loss.retrieval import InBatchLogQLoss


def make_loss(tmp_path, counts, coefficient):
    counts_file = tmp_path / 'target_counts.pkl'
    with counts_file.open('wb') as destination:
        pickle.dump(counts, destination)
    return InBatchLogQLoss(
        'queries',
        'positives',
        'items',
        str(counts_file),
        user_ids_prefix='users',
        logq_lambda=coefficient,
    ).double()


@pytest.mark.parametrize('anchor', range(6))
def test_masked_without_replacement_proposal_matches_exact_enumeration(tmp_path, anchor):
    users = [1, 1, 2, 3, 4, 4]
    items = [1, 2, 2, 3, 2, 3]
    population = len(items)
    batch_size = 3
    marginal = Counter(items)
    loss = make_loss(tmp_path, [0, 1, 3, 2, 0], coefficient=1)
    batches = list(itertools.combinations([row for row in range(population) if row != anchor], batch_size - 1))
    retained_counts = Counter()
    for other_rows in batches:
        row_indices = [anchor, *other_rows]
        candidate_items = torch.tensor([items[row] for row in row_indices])
        candidate_users = torch.tensor([users[row] for row in row_indices])
        invalid = loss._invalid_mask({'users': candidate_users}, candidate_items)
        assert invalid[0, 0].item() is False
        for column, row in enumerate(other_rows, start=1):
            permitted = users[row] != users[anchor] and items[row] != items[anchor]
            assert invalid[0, column].item() is (not permitted)
            if permitted:
                retained_counts[items[row]] += 1
    own_user_counts = Counter(item for item, user in zip(items, users) if user == users[anchor])
    for item in marginal:
        expected = (
            0
            if item == items[anchor]
            else (batch_size - 1) * (marginal[item] - own_user_counts[item]) / (population - 1)
        )
        assert retained_counts[item] / len(batches) == pytest.approx(expected)
    assert sum(retained_counts.values()) / len(batches) < batch_size - 1


@pytest.mark.parametrize('counts', [[0, 1, 1, 1, 0], [0, 1, 3, 6, 0]])
@pytest.mark.parametrize('coefficient', [0.0, 0.4, 1.0])
def test_negatives_only_is_relative_popularity_plus_global_margin(tmp_path, counts, coefficient):
    queries = torch.tensor([[0.3, -0.4], [0.7, 0.2], [-0.2, 0.9]], dtype=torch.float64, requires_grad=True)
    positives = torch.tensor([[0.2, 0.8], [-0.3, 0.5], [0.6, -0.1]], dtype=torch.float64, requires_grad=True)
    loss = make_loss(tmp_path, counts, coefficient)
    inputs = {
        'queries': queries,
        'positives': positives,
        'items': torch.tensor([1, 2, 3]),
        'users': torch.tensor([7, 8, 9]),
    }
    actual = loss(inputs)
    vocabulary_size = 3
    terms = []
    for row in range(3):
        positive_score = torch.dot(queries[row], positives[row])
        candidate_scores = [positive_score]
        for column in range(3):
            if column != row:
                probability = counts[column + 1] / sum(counts)
                relative_popularity = -coefficient * math.log(vocabulary_size * probability)
                global_margin = coefficient * math.log(vocabulary_size)
                candidate_scores.append(
                    torch.dot(queries[row], positives[column]) + relative_popularity + global_margin
                )
        terms.append(torch.logsumexp(torch.stack(candidate_scores), 0) - positive_score)
    expected = torch.stack(terms).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for observed, reference in zip(
        torch.autograd.grad(actual, (queries, positives)), torch.autograd.grad(expected, (queries, positives))
    ):
        torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-7)


def test_uniform_catalogue_changes_objective_even_with_no_popularity_imbalance(tmp_path):
    vocabulary_size = 10
    inputs = {
        'queries': torch.zeros(3, 2),
        'positives': torch.zeros(3, 2),
        'items': torch.tensor([1, 2, 3]),
        'users': torch.tensor([1, 2, 3]),
    }
    counts = [0, *([1] * vocabulary_size), 0]
    baseline = make_loss(tmp_path, counts, coefficient=0)(inputs)
    corrected = make_loss(tmp_path, counts, coefficient=1)(inputs)
    assert baseline.item() == pytest.approx(math.log(3))
    assert corrected.item() == pytest.approx(math.log(1 + 2 * vocabulary_size))
