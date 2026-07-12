"""Reference tests for the ranking metrics.

Every metric is checked against a hand-computed naive reference on a tricky
multi-target batch (variable target counts, offset bookkeeping, zero-target
user). Run directly (`python tests/test_metrics.py`) or via pytest.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from irec.metric.base import (  # noqa: E402
    ComiRecNDCGMetric,
    MCLSRHitRateMetric,
    MCLSRNDCGMetric,
    MCLSRRecallMetric,
)

# Batch of 4 users, k=5.
#   user 0: targets {7, 8, 9};  hits at ranks 1 (item 7) and 4 (item 8)
#   user 1: targets {13};       hit at rank 0
#   user 2: targets {21, 22};   no hits
#   user 3: targets {} (zero targets)
PREDS = torch.tensor([
    [10, 7, 11, 12, 8],
    [13, 14, 15, 16, 17],
    [30, 31, 32, 33, 34],
    [40, 41, 42, 43, 44],
])
INPUTS = {
    'p': PREDS,
    'labels.ids': torch.tensor([7, 8, 9, 13, 21, 22]),
    'labels.length': torch.tensor([3, 1, 2, 0]),
}

W = [1 / math.log2(r + 2) for r in range(5)]  # rank discounts, rank 0-based


def test_recall():
    got = MCLSRRecallMetric(k=5)(INPUTS, 'p', 'labels')
    want = [2 / 3, 1 / 1, 0 / 2, 0.0]
    assert len(got) == 4
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-6, (got, want)


def test_hitrate():
    got = MCLSRHitRateMetric(k=5)(INPUTS, 'p', 'labels')
    assert got == [1.0, 1.0, 0.0, 0.0], got


def test_ndcg_standard():
    got = MCLSRNDCGMetric(k=5)(INPUTS, 'p', 'labels')
    # user 0: dcg = W[1] + W[4]; idcg over min(k, 3 targets) = W[0] + W[1] + W[2]
    want0 = (W[1] + W[4]) / (W[0] + W[1] + W[2])
    # user 1: dcg = W[0]; idcg over min(k, 1) = W[0]
    want1 = 1.0
    want = [want0, want1, 0.0, 0.0]
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-5, (got, want)


def test_ndcg_comirec():
    got = ComiRecNDCGMetric(k=5)(INPUTS, 'p', 'labels')
    # user 0: idcg over the ACTUAL number of hits (2) = W[0] + W[1]
    want0 = (W[1] + W[4]) / (W[0] + W[1])
    want = [want0, 1.0, 0.0, 0.0]
    for g, w in zip(got, want):
        assert abs(g - w) < 1e-5, (got, want)
    # ComiRec normalization is never below the standard one
    std = MCLSRNDCGMetric(k=5)(INPUTS, 'p', 'labels')
    for c, s in zip(got, std):
        assert c >= s - 1e-9


def test_offset_bookkeeping_no_leakage():
    # user 1's prediction 13 must NOT count as a hit for user 0 and vice versa
    inputs = {
        'p': torch.tensor([[13, 1, 2, 3, 4], [7, 1, 2, 3, 4]]),
        'labels.ids': torch.tensor([7, 13]),  # user0 -> {7}, user1 -> {13}
        'labels.length': torch.tensor([1, 1]),
    }
    got = MCLSRRecallMetric(k=5)(inputs, 'p', 'labels')
    assert got == [0.0, 0.0], got


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
    print('\nALL METRIC TESTS PASSED')
