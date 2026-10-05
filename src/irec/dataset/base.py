"""Dataset base classes: the registry root and the sequence-file readers shared by the MCLSR and SASRec datasets."""

from irec.utils import MetaParent
import logging

logger = logging.getLogger(__name__)


class BaseDataset(metaclass=MetaParent):
    """Registry root of the datasets: samplers for the three splits and the catalogue sizes."""

    def get_samplers(self):
        raise NotImplementedError

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length


class BaseSequenceDataset(BaseDataset):
    """A dataset made of user sequences: holds the three samplers and reads ``user item item ...`` files."""

    def __init__(
        self,
        train_sampler,
        validation_sampler,
        test_sampler,
        num_users,
        num_items,
        max_sequence_length,
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @staticmethod
    def _create_sequences(data, max_sample_len=None):  # TODO
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]]
            if max_sample_len is not None:
                item_ids = item_ids[-max_sample_len:]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_length,
        )

    def get_samplers(self):
        return (
            self._train_sampler,
            self._validation_sampler,
            self._test_sampler,
        )

    @property
    def meta(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length,
        }
