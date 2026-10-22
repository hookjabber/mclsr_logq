"""Negative sampler base class: draws item ids for an example, excluding what the example contains."""

from collections import defaultdict
from numbers import Integral

from irec.utils import MetaParent


class BaseNegativeSampler(metaclass=MetaParent):
    def __init__(self, dataset, num_users, num_items):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

        self._seen_items = defaultdict(set)
        for sample in self._dataset:
            user_id = sample['user.ids'][0]
            items = list(sample['item.ids'])
            if any(
                not isinstance(item, Integral) or isinstance(item, bool) or not 1 <= item <= num_items for item in items
            ):
                raise ValueError('negative sampler item IDs must be non-padding catalogue IDs')
            self._seen_items[user_id].update(items)

    @staticmethod
    def validate_request(num_negatives, available):
        if isinstance(num_negatives, bool) or not isinstance(num_negatives, Integral) or num_negatives < 0:
            raise ValueError('num_negatives must be a nonnegative integer')
        if num_negatives > available:
            raise ValueError(f'requested {num_negatives} unique negatives, but only {available} are available')

    def _sample_seen_items(self, sample):
        items = sample['item.ids']
        if any(
            not isinstance(item, Integral) or isinstance(item, bool) or not 1 <= item <= self._num_items
            for item in items
        ):
            raise ValueError('negative sampler item IDs must be non-padding catalogue IDs')
        return self._seen_items[sample['user.ids'][0]].union(items)

    def generate_negative_samples(self, sample, num_negatives):
        raise NotImplementedError
