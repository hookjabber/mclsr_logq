"""The paired statistics behind results/stats: interval coverage and the sign-flip p-value."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paired_bootstrap import paired_stats  # noqa: E402


def test_clear_effect_and_null():
    rng = np.random.default_rng(0)
    users, seeds = 2000, 3
    effect = 0.02 + rng.normal(0, 0.05, size=(users, seeds))  # per-user differences around +0.02
    obs, lo, hi, p = paired_stats(effect, np.random.default_rng(1), draws=2000)
    assert abs(obs - effect.mean()) < 1e-12
    assert lo < 0.02 < hi and hi - lo < 0.01 and p < 0.001
    null = rng.normal(0, 0.05, size=(users, seeds))
    obs, lo, hi, p = paired_stats(null, np.random.default_rng(1), draws=2000)
    assert lo < 0 < hi and p > 0.05


def test_interval_tracks_sample_size():
    rng = np.random.default_rng(0)
    wide = paired_stats(rng.normal(0.01, 0.1, size=(200, 3)), np.random.default_rng(1), draws=2000)
    narrow = paired_stats(rng.normal(0.01, 0.1, size=(5000, 3)), np.random.default_rng(1), draws=2000)
    assert (wide[2] - wide[1]) > 3 * (narrow[2] - narrow[1])
