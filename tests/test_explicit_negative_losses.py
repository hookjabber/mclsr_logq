"""Reference tests for the two losses that use explicit sampled negatives (arms 01_uniform*, 01_popular127,
02_popular127_logq): the logQ-corrected sampled softmax with an item table, checked against a hand-written
cross-entropy with the same masking and correction."""

import os
import pickle
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from irec.loss import MCLSRLogqLoss, SamplesSoftmaxLoss  # noqa: E402


def _counts_file(counts):
    path = os.path.join(tempfile.mkdtemp(), 'counts.pkl')
    with open(path, 'wb') as f:
        pickle.dump(counts, f)
    return path


def _inputs(seed=0, batch=6, num_neg=5, dim=8, num_items=12):
    g = torch.Generator().manual_seed(seed)
    queries = torch.randn(batch, dim, generator=g)
    table = torch.randn(num_items + 2, dim, generator=g)
    pos_ids = torch.randint(1, num_items + 1, (batch,), generator=g)
    neg_ids = torch.randint(1, num_items + 1, (batch, num_neg), generator=g)
    neg_ids[0, 0] = pos_ids[0]  # one accidental hit: the target sampled as its own negative
    return {
        'q': queries,
        'pos': table[pos_ids],
        'neg': table[neg_ids],
        'pos_ids': pos_ids,
        'neg_ids': neg_ids,
    }, table


def _manual(inputs, log_q, lam):
    q, pos, neg, pos_ids, neg_ids = (inputs[k] for k in ('q', 'pos', 'neg', 'pos_ids', 'neg_ids'))
    s_pos = (q * pos).sum(-1, keepdim=True) - lam * log_q[pos_ids][:, None]
    s_neg = torch.einsum('bd,bnd->bn', q, neg) - lam * log_q[neg_ids]
    s_neg = s_neg.masked_fill(neg_ids == pos_ids[:, None], -1e12)
    return -torch.log_softmax(torch.cat([s_pos, s_neg], dim=1), dim=1)[:, 0].mean()


def test_explicit_negatives_logq_matches_manual():
    counts = [0, 5, 1, 7, 2, 9, 4, 1, 3, 6, 2, 8, 1, 0]  # 12 items + pad + mask
    probs = torch.clamp(torch.tensor(counts, dtype=torch.float32) / sum(counts), min=1e-10)
    log_q = torch.log(probs)
    for lam in (0.0, 1.0):
        loss = MCLSRLogqLoss(
            queries_prefix='q',
            positive_prefix='pos',
            negative_prefix='neg',
            positive_ids_prefix='pos_ids',
            negative_ids_prefix='neg_ids',
            path_to_item_counts=_counts_file(counts),
            logq_lambda=lam,
        )
        inputs, _ = _inputs()
        got = loss(inputs)
        want = _manual(inputs, log_q, lam)
        assert torch.allclose(got, want, atol=1e-6), (lam, float(got), float(want))


def test_explicit_negatives_lambda_zero_is_plain_sampled_softmax():
    counts = [0] + [3] * 12 + [0]
    loss = MCLSRLogqLoss(
        queries_prefix='q',
        positive_prefix='pos',
        negative_prefix='neg',
        positive_ids_prefix='pos_ids',
        negative_ids_prefix='neg_ids',
        path_to_item_counts=_counts_file(counts),
        logq_lambda=0.0,
    )
    inputs, _ = _inputs(seed=3)
    got = loss(inputs)
    # with lambda = 0 the count table must not matter at all
    other = MCLSRLogqLoss(
        queries_prefix='q',
        positive_prefix='pos',
        negative_prefix='neg',
        positive_ids_prefix='pos_ids',
        negative_ids_prefix='neg_ids',
        path_to_item_counts=_counts_file([0] + list(range(1, 13)) + [0]),
        logq_lambda=0.0,
    )(inputs)
    assert torch.allclose(got, other, atol=1e-7)


def test_plain_sampled_softmax_matches_manual_and_has_no_accidental_hit_mask():
    # the framework's uncorrected sampled softmax (uniform-negative arms): cross-entropy over
    # [positive, negatives] with NO accidental-hit masking — documented in RESULTS.md
    loss = SamplesSoftmaxLoss(queries_prefix='q', positive_prefix='pos', negative_prefix='neg')
    inputs, _ = _inputs(seed=5)
    got = loss(inputs)
    q, pos, neg = inputs['q'], inputs['pos'], inputs['neg']
    s_pos = (q * pos).sum(-1, keepdim=True)
    s_neg = torch.einsum('bd,bnd->bn', q, neg)
    want = -torch.log_softmax(torch.cat([s_pos, s_neg], dim=1), dim=1)[:, 0].mean()
    assert torch.allclose(got, want, atol=1e-6), (float(got), float(want))
    # row 0 contains the target among its negatives; masking it would lower the loss
    masked = s_neg.clone()
    masked[0, 0] = -1e12
    want_masked = -torch.log_softmax(torch.cat([s_pos, masked], dim=1), dim=1)[:, 0].mean()
    assert want_masked < want, 'the accidental hit in row 0 should matter'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
    print('\nALL EXPLICIT-NEGATIVE TESTS PASSED')
