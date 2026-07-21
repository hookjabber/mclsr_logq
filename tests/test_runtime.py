"""Smoke tests for the training-loop and analysis runtime paths.

Covers the round-6/7 audit fixes: exact step limit, multi-metric best
tracking, checkpoint unwrapping, SASRec negative rejection sampling, and the
decile-analysis helpers (tie-preserving bins, Holm adjustment).

    python tests/test_runtime.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'),
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'),
)

from irec.train import train, unwrap_state_dict  # noqa: E402


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        return {}


class _CountingOptimizer:
    def __init__(self):
        self.steps = 0

    def step(self, loss):
        self.steps += 1


def _batches(n):
    return [{'x': torch.zeros(1)} for _ in range(n)]


def test_step_limit_is_exact():
    optimizer = _CountingOptimizer()
    train(
        dataloader=_batches(100),
        model=_TinyModel(),
        optimizer=optimizer,
        loss_function=lambda batch: torch.zeros(1),
        callback=lambda batch, step: None,
        epoch_cnt=50,
        step_cnt=7,
    )
    assert optimizer.steps == 7, optimizer.steps


def test_multi_metric_best_tracking():
    curves = {
        'validation/a': [0.1, 0.5, 0.5, 0.3],  # first max at step 1
        'validation/b': [0.9, 0.1, 0.2, 0.95],  # max at step 3
    }
    seen = []

    def callback(batch, step):
        for name, values in curves.items():
            batch[name] = values[step]
        seen.append(step)

    model = _TinyModel()
    states = {}

    real_deepcopy_marker = {}

    class _MarkedModel(_TinyModel):
        def state_dict(self, *a, **kw):
            return {'step': len(seen) - 1}

    model = _MarkedModel()
    best = train(
        dataloader=_batches(4),
        model=model,
        optimizer=_CountingOptimizer(),
        loss_function=lambda batch: torch.zeros(1),
        callback=callback,
        epoch_cnt=1,
        best_metric=list(curves),
    )
    assert best['validation/a']['step'] == 1, best  # strict: tie keeps FIRST
    assert best['validation/b']['step'] == 3, best
    assert best['validation/a']['value'] == 0.5
    assert best['validation/b']['value'] == 0.95
    assert best['validation/a']['state'] == {'step': 1}
    del states, real_deepcopy_marker


def test_unwrap_state_dict():
    bare = {'w': torch.ones(1)}
    assert unwrap_state_dict(bare) is bare
    assert unwrap_state_dict({'model_state_dict': bare}) is bare
    assert unwrap_state_dict({'model': bare, 'optimizer': {}}) is bare


def test_sasrec_negative_rejection():
    from irec.models.sasrec import SasRecModel
    torch.manual_seed(0)
    num_items = 12
    model = SasRecModel(
        sequence_prefix='item',
        positive_prefix='labels',
        num_items=num_items,
        max_sequence_length=8,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    model.train()
    # two users; user A consumed 1..6 (positives 2..7), user B consumed 8..10
    seq = torch.tensor([1, 2, 3, 4, 5, 6, 8, 9, 10])
    lengths = torch.tensor([6, 3])
    positives = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10, 11])
    for trial in range(5):
        out = model({
            'item.ids': seq, 'item.length': lengths, 'labels.ids': positives,
        })
        negatives = out['negative_ids']
        assert negatives.shape == positives.shape
        assert (negatives >= 1).all() and (negatives <= num_items).all()
        # forbidden = the user's FULL sequence: inputs UNION positives — the
        # final target (7 / 11) is excluded for EVERY query of that user
        forbidden_a = set(seq[:6].tolist()) | {7}
        forbidden_b = set(seq[6:].tolist()) | {11}
        for i in range(9):
            forbidden = forbidden_a if i < 6 else forbidden_b
            assert negatives[i].item() not in forbidden, (i, negatives[i])
            assert negatives[i].item() != positives[i].item()

    # dense catalog: user consumed everything -> must raise, never silently
    # accept a colliding negative
    dense = SasRecModel(
        sequence_prefix='item', positive_prefix='labels', num_items=2,
        max_sequence_length=4, embedding_dim=8, num_heads=2, num_layers=1,
        dim_feedforward=16,
    )
    dense.train()
    try:
        dense({
            'item.ids': torch.tensor([1]), 'item.length': torch.tensor([1]),
            'labels.ids': torch.tensor([2]),
        })
        raise AssertionError('dense-catalog rejection must raise')
    except RuntimeError:
        pass


def test_tie_preserving_bins_and_holm():
    from decile_recall import build_bins, holm_adjust
    # ids 1..10 real; heavy ties at count 5
    counts = np.array([1, 5, 5, 5, 5, 2, 2, 9, 9, 3, 7, 1], dtype=float)
    bins, n_bins = build_bins(counts, num_bins=4, tie_preserving=True)
    real = counts[1:11]
    for value in np.unique(real):
        assigned = {bins[i + 1] for i in range(10) if real[i] == value}
        assert len(assigned) == 1, (value, assigned)  # ties never split
    order_means = [
        np.mean([real[i] for i in range(10) if bins[i + 1] == b])
        for b in range(n_bins)
    ]
    assert order_means == sorted(order_means)

    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
    assert np.isclose(adjusted[0], 0.04)   # 0.01 * 4
    assert np.isclose(adjusted[3], 0.5)
    assert (np.asarray(adjusted) >= [0.01, 0.04, 0.03, 0.5]).all()
    assert adjusted[1] >= adjusted[2] >= adjusted[0]  # monotone step-down


def test_scientific_split_before_truncation():
    from irec.dataset.base import BaseSequenceDataset
    lines = ['7 ' + ' '.join(str(i) for i in range(1, 11))]  # 10 items
    _, seqs, _, _, _ = BaseSequenceDataset._create_sequences(lines)
    assert seqs[0] == list(range(1, 11))  # no truncation without max_len
    _, seqs, _, _, _ = BaseSequenceDataset._create_sequences(lines, 4)
    assert seqs[0] == [7, 8, 9, 10]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
    print('\nALL RUNTIME TESTS PASSED')
