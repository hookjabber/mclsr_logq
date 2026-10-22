import itertools
import json
import math
from pathlib import Path
import pickle
import runpy
import sys

import pytest
import torch

from irec.loss.contrastive import FpsLogQLoss, FpsLoss


ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ('cross_only', 'paper', 'symmetric')


def _counts_file(tmp_path, counts):
    path = tmp_path / 'counts.pkl'
    with path.open('wb') as destination:
        pickle.dump(counts, destination)
    return str(path)


def _loss(path, **options):
    return FpsLogQLoss('first', 'second', 'ids', path, tau=0.7, **options)


def _embeddings():
    first = torch.tensor([[0.2, 0.8], [-0.4, 0.6], [0.9, -0.3]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[0.7, -0.1], [0.3, 0.5], [-0.6, 0.4]], dtype=torch.float64, requires_grad=True)
    return first, second


def _scalar_objective(first, second, ids, counts, scheme, mode, mask_duplicates, normalize):
    views = (first, second)
    if normalize:
        views = tuple(view / view.square().sum(dim=1, keepdim=True).sqrt() for view in views)
    terms = []
    anchor_views = (0, 1) if scheme == 'symmetric' else (0,)
    for anchor_view in anchor_views:
        for anchor_row, anchor_id in enumerate(ids.tolist()):
            anchor = views[anchor_view][anchor_row]
            positive = views[1 - anchor_view][anchor_row]
            positive_score = (anchor * positive).sum() / 0.7
            candidate_scores = [positive_score]
            candidate_views = (1 - anchor_view,) if scheme == 'cross_only' else (0, 1)
            for candidate_view in candidate_views:
                for candidate_row, candidate_id in enumerate(ids.tolist()):
                    if candidate_row == anchor_row:
                        continue
                    if mask_duplicates and candidate_id == anchor_id:
                        continue
                    probability = counts[candidate_id] / sum(counts)
                    if mode == 'inbatch_negative':
                        probability = -math.expm1((len(ids) - 1) * math.log1p(-probability))
                    score = (anchor * views[candidate_view][candidate_row]).sum() / 0.7
                    candidate_scores.append(score - 0.6 * math.log(probability))
            terms.append(torch.logsumexp(torch.stack(candidate_scores), dim=0) - positive_score)
    objective = torch.stack(terms).mean()
    return objective / 2 if scheme == 'symmetric' else objective


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('mode', ('sample', 'inbatch_negative'))
@pytest.mark.parametrize('mask_duplicates', (False, True))
@pytest.mark.parametrize('normalize', (False, True))
def test_values_and_both_gradients_match_scalar_catalogue_oracle(tmp_path, scheme, mode, mask_duplicates, normalize):
    counts = [0, 3, 11, 4, 2, 5, 7, 1]
    ids = torch.tensor([6, 2, 6])
    first, second = _embeddings()
    loss = _loss(
        _counts_file(tmp_path, counts),
        scheme=scheme,
        logq_lambda=0.6,
        logq_probability_mode=mode,
        mask_false_negatives=mask_duplicates,
        normalize_embeddings=normalize,
    ).double()
    actual = loss({'first': first, 'second': second, 'ids': ids})
    expected = _scalar_objective(first, second, ids, counts, scheme, mode, mask_duplicates, normalize)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
    actual_gradients = torch.autograd.grad(actual, (first, second))
    expected_gradients = torch.autograd.grad(expected, (first, second))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('mask_duplicates', (False, True))
def test_zero_lambda_equals_fps_when_masks_do_not_remove_other_samples(tmp_path, scheme, mask_duplicates):
    ids = torch.tensor([1, 2, 3] if mask_duplicates else [1, 1, 3])
    first, second = _embeddings()
    inputs = {'first': first, 'second': second, 'ids': ids}
    corrected = _loss(
        _counts_file(tmp_path, [0, 1, 2, 3]),
        scheme=scheme,
        logq_lambda=0,
        mask_false_negatives=mask_duplicates,
    )(inputs)
    plain = FpsLoss('first', 'second', tau=0.7, scheme=scheme)(inputs)
    torch.testing.assert_close(corrected, plain, atol=1e-12, rtol=1e-12)
    for corrected_gradient, plain_gradient in zip(
        torch.autograd.grad(corrected, (first, second)), torch.autograd.grad(plain, (first, second))
    ):
        torch.testing.assert_close(corrected_gradient, plain_gradient, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_zero_lambda_default_duplicate_mask_changes_objective(tmp_path, scheme):
    inputs = {'first': torch.zeros(3, 2), 'second': torch.zeros(3, 2), 'ids': torch.tensor([1, 1, 2])}
    actual = _loss(_counts_file(tmp_path, [0, 2, 1]), scheme=scheme, logq_lambda=0)(inputs)
    plain = FpsLoss('first', 'second', tau=0.7, scheme=scheme)(inputs)
    candidates_per_anchor = (2, 2, 3) if scheme == 'cross_only' else (3, 3, 5)
    scale = 0.5 if scheme == 'symmetric' else 1.0
    expected = scale * sum(math.log(count) for count in candidates_per_anchor) / 3
    assert actual.item() == pytest.approx(expected)
    assert actual.item() < plain.item()


@pytest.mark.parametrize('use_mean', (False, True))
@pytest.mark.parametrize('corrected', (False, True))
def test_symmetric_scaling_and_paper_direction_are_not_interchangeable(tmp_path, use_mean, corrected):
    first, second = _embeddings()
    path = _counts_file(tmp_path, [0, 1, 2, 3])

    def evaluate(scheme, first_view, second_view):
        inputs = {'first': first_view, 'second': second_view, 'ids': torch.tensor([1, 2, 3])}
        if corrected:
            return _loss(path, scheme=scheme, use_mean=use_mean)(inputs)
        return FpsLoss('first', 'second', tau=0.7, scheme=scheme, use_mean=use_mean)(inputs)

    paper_forward = evaluate('paper', first, second)
    paper_reverse = evaluate('paper', second, first)
    symmetric = evaluate('symmetric', first, second)
    multiplier = 0.25 if use_mean else 0.5
    torch.testing.assert_close(symmetric, multiplier * (paper_forward + paper_reverse))
    assert not torch.isclose(paper_forward, paper_reverse)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_uniform_q_changes_negative_positive_margin_because_positive_is_not_corrected(tmp_path, scheme):
    inputs = {'first': torch.zeros(3, 2), 'second': torch.zeros(3, 2), 'ids': torch.tensor([1, 2, 3])}
    path = _counts_file(tmp_path, [0, 1, 1, 1])
    corrected = _loss(path, scheme=scheme, logq_probability_mode='sample')(inputs)
    centered = _loss(path, scheme=scheme, logq_probability_mode='sample', center_log_q=True)(inputs)
    baseline = _loss(path, scheme=scheme, logq_lambda=0)(inputs)
    negatives = 2 if scheme == 'cross_only' else 4
    scale = 0.5 if scheme == 'symmetric' else 1.0
    assert corrected.item() == pytest.approx(scale * math.log(1 + 3 * negatives))
    assert corrected.item() > baseline.item()
    torch.testing.assert_close(centered, baseline)


def test_row_multiplicity_is_not_entity_inclusion_probability(tmp_path):
    loss = _loss(_counts_file(tmp_path, [0, 2, 1, 1]))
    draws = list(itertools.product((1, 1, 2, 3), repeat=2))
    expected_multiplicity = sum(batch.count(1) for batch in draws) / len(draws)
    inclusion_probability = sum(1 in batch for batch in draws) / len(draws)
    implemented_q = loss._sample_log_q(torch.tensor([1]), 2).exp().item()
    assert implemented_q == pytest.approx(inclusion_probability)
    assert expected_multiplicity == 1.0
    assert expected_multiplicity / implemented_q == pytest.approx(4 / 3)
    assert inclusion_probability / implemented_q == pytest.approx(1.0)


def test_leave_own_out_is_not_probability_of_masked_unresampled_negatives(tmp_path):
    loss = _loss(_counts_file(tmp_path, [0, 2, 1, 1]), leave_own_out=True)
    negative_batches = list(itertools.product((1, 1, 2, 3), repeat=2))
    masked_batches = [tuple(candidate for candidate in batch if candidate != 1) for batch in negative_batches]
    observed_inclusion = sum(2 in batch for batch in masked_batches) / len(masked_batches)
    implemented_q = loss._loo_log_q_square(torch.tensor([1, 2]), 2)[0, 1].exp().item()
    assert observed_inclusion == 7 / 16
    assert implemented_q == pytest.approx(3 / 4)
    assert implemented_q != pytest.approx(observed_inclusion)


def test_verified_train_line_denominator_avoids_absent_user_pseudocount_mass(tmp_path):
    path = _counts_file(tmp_path, [1, 3, 2, 1])
    candidate_ids = torch.tensor([2, 1])
    exact_marginal = _loss(path, logq_probability_mode='sample', counts_denominator=5)
    smoothed = _loss(path, logq_probability_mode='sample')
    actual = exact_marginal._sample_log_q(candidate_ids, 2).exp()
    torch.testing.assert_close(actual, torch.tensor([2 / 5, 3 / 5]))
    assert not torch.allclose(actual, smoothed._sample_log_q(candidate_ids, 2).exp())


@pytest.fixture
def role_artifacts(tmp_path, monkeypatch):
    source = tmp_path / 'train_mclsr.txt'
    source.write_text('1 1 1 2\n2 1 3 4\n3 2 3 4\n4 2 4 3\n')
    (tmp_path / 'preprocessing.json').write_text(json.dumps({'protocol': 'windowed-v2', 'max_history_length': 2}))
    target = tmp_path / 'target.pkl'
    context = tmp_path / 'context.pkl'
    script = ROOT / 'scripts' / 'generate_mclsr_role_counts.py'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            str(script),
            '--input',
            str(source),
            '--num_items',
            '4',
            '--target_output',
            str(target),
            '--context_output',
            str(context),
        ],
    )
    runpy.run_path(str(script), run_name='__main__')
    with target.open('rb') as stream:
        targets = pickle.load(stream)
    with context.open('rb') as stream:
        contexts = pickle.load(stream)
    return targets, contexts, str(context)


def test_role_counts_are_target_events_and_unique_context_presence_per_line(role_artifacts):
    targets, contexts, _ = role_artifacts
    assert list(targets['counts']) == [0, 0, 1, 1, 2, 0]
    assert list(contexts['counts']) == [0, 2, 2, 2, 1, 0]
    assert targets['denominator'] == contexts['denominator'] == 4
    assert targets['max_len'] == contexts['max_len'] == 3
    assert contexts['counts'].sum() / contexts['denominator'] == 7 / 4


def test_context_q_iid_transform_is_not_exact_shuffled_batch_inclusion(role_artifacts):
    _, _, path = role_artifacts
    loss = _loss(path, expected_counts_role='context-inclusion', expected_counts_max_len=3, num_draws_value=2)
    contexts = ({1}, {1, 3}, {2, 3}, {2, 4})
    unions = [contexts[first] | contexts[second] for first, second in itertools.combinations(range(4), 2)]
    exact_inclusion = sum(1 in union for union in unions) / len(unions)
    iid_inclusion = loss._sample_log_q(torch.tensor([1]), 2).exp().item()
    assert exact_inclusion == 5 / 6
    assert iid_inclusion == pytest.approx(3 / 4)
    assert iid_inclusion != pytest.approx(exact_inclusion)


def test_line_marginals_cannot_identify_anchor_conditional_item_q():
    joint_contexts = ({1, 2}, {1, 2}, {3}, {3})
    disjoint_contexts = ({1, 3}, {1, 3}, {2}, {2})
    joint_marginals = [sum(item in row for row in joint_contexts) for item in (1, 2, 3)]
    disjoint_marginals = [sum(item in row for row in disjoint_contexts) for item in (1, 2, 3)]
    assert joint_marginals == disjoint_marginals == [2, 2, 2]
    assert sum(2 in row for row in joint_contexts if 1 in row) / 2 == 1
    assert sum(2 in row for row in disjoint_contexts if 1 in row) / 2 == 0


@pytest.mark.parametrize('method', ('_sample_log_q', '_loo_log_q_square', '_candidate_log_q_leave_own_out'))
def test_small_nonzero_inclusion_q_matches_stable_probability(tmp_path, method):
    loss = _loss(_counts_file(tmp_path, [0, 1, 50_000_000, 49_999_999]), leave_own_out=method != '_sample_log_q')
    ids = torch.tensor([2, 1])
    probability = float(loss._prob_table[1])
    if method != '_sample_log_q':
        probability /= 1 - float(loss._prob_table[2])
    expected = -math.expm1(127 * math.log1p(-probability))
    probabilities = getattr(loss, method)(ids, 127).exp()
    actual = probabilities[1].item() if method == '_sample_log_q' else probabilities[0, 1].item()
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-12)


@pytest.mark.parametrize('method', ('_sample_log_q', '_loo_log_q_square', '_candidate_log_q_leave_own_out'))
@pytest.mark.parametrize('draws', (-2, 0, 1, 127))
def test_certain_candidate_preserves_zero_draw_floor_and_positive_draw_certainty(tmp_path, method, draws):
    loss = _loss(_counts_file(tmp_path, [0, 1]), leave_own_out=method != '_sample_log_q')
    actual = getattr(loss, method)(torch.tensor([1]), draws)
    expected = math.log(1e-10) if draws <= 0 else 0.0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.full_like(actual, expected))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('draws', (-2, 0, 1, 127))
def test_stable_inclusion_helper_preserves_floor_dtype_and_boundaries(dtype, draws):
    probabilities = torch.tensor([0.0, 1e-12, 1e-8, 0.25, 1.0], dtype=dtype)
    actual = FpsLogQLoss._inclusion_log_q(probabilities, draws)
    expected = []
    for probability in probabilities.tolist():
        if draws <= 0:
            inclusion = 0.0
        elif probability == 1.0:
            inclusion = 1.0
        else:
            inclusion = -math.expm1(draws * math.log1p(-probability))
        expected.append(math.log(max(inclusion, 1e-10)))
    assert actual.dtype == dtype
    assert actual.device == probabilities.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype))
