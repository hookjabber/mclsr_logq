"""Uniform unique negatives with bounded rejection and a conditional fallback."""

from irec.dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):
    """Uniform negatives over the catalogue, rejecting the items of the example."""

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )

    def generate_negative_samples(self, sample, num_negatives):
        """Reject seen IDs, then sample the remaining complement if rejection stalls."""
        seen = self._sample_seen_items(sample)
        self.validate_request(num_negatives, self._num_items - len(seen))
        negatives = set()
        attempts = 0
        while len(negatives) < num_negatives and attempts < max(32, 4 * num_negatives):
            negative_idx = np.random.randint(1, self._num_items + 1)
            attempts += 1
            if negative_idx not in seen:
                negatives.add(negative_idx)
        if len(negatives) < num_negatives:
            remaining = [item for item in range(1, self._num_items + 1) if item not in seen and item not in negatives]
            negatives.update(np.random.choice(remaining, size=num_negatives - len(negatives), replace=False))
        return list(negatives)
