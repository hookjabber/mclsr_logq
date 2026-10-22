"""Popularity-proportional unique negatives with bounded rejection."""

import numpy as np
from irec.dataset.negative_samplers.base import BaseNegativeSampler
from collections import Counter


class PopularNegativeSampler(BaseNegativeSampler, config_name='popular'):
    """Negatives drawn proportionally to training popularity, rejecting the items of the example."""

    def __init__(self, dataset, num_users, num_items):
        super().__init__(
            dataset=dataset,
            num_users=num_users,
            num_items=num_items,
        )

        self._item_ids, self._probs = self._calculate_item_probabilities()
        self._support_items = set(self._item_ids.tolist())

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )

    def _calculate_item_probabilities(self):
        """
        Calculates sampling probabilities proportional to item popularity.
        This distribution is required to provide non-zero p_j values for LogQ correction.
        """
        counts = Counter()
        for sample in self._dataset:
            for item_id in sample['item.ids']:
                counts[item_id] += 1

        items = np.array(list(counts.keys()))
        freqs = np.array(list(counts.values()), dtype=np.float64)
        probabilities = freqs / freqs.sum() if len(freqs) else freqs

        return items, probabilities

    def generate_negative_samples(self, sample, num_negatives):
        """
        Stochastic sampling proportional to popularity.

        For LogQ correction (Yi et al., 2019), we need a stochastic
        sampling process where p_j > 0 for all items in the distribution.
        """
        seen = self._sample_seen_items(sample)
        available = len(self._support_items) - len(self._support_items.intersection(seen))
        self.validate_request(num_negatives, available)
        negatives = set()
        attempts = 0
        budget = max(32, 4 * num_negatives)
        while len(negatives) < num_negatives and attempts < budget:
            draw_count = min(num_negatives - len(negatives), budget - attempts)
            sampled_ids = np.random.choice(self._item_ids, size=draw_count, p=self._probs, replace=True)
            attempts += draw_count

            for idx in sampled_ids:
                if idx not in seen:
                    negatives.add(idx)
                    if len(negatives) == num_negatives:
                        break
        if len(negatives) < num_negatives:
            eligible = np.array([item not in seen and item not in negatives for item in self._item_ids])
            probabilities = self._probs[eligible]
            probabilities = probabilities / probabilities.sum()
            negatives.update(
                np.random.choice(
                    self._item_ids[eligible], size=num_negatives - len(negatives), replace=False, p=probabilities
                )
            )
        return list(negatives)
