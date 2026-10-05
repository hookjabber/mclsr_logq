"""Sequence datasets of the study.

* ``MCLSRDataset`` (``mclsr``): the prefix ladder ``train_mclsr.txt`` (one training example per target
  position) and the validation / test histories and targets, as written by ``scripts/prepare_dataset.py``;
* ``SASRecDataset`` (``sasrec_comparison``): the same evaluation sets with ``train_sasrec.txt`` (one full
  cropped sequence per user, every position a query) or the ladder as the training file;
* ``SequenceDataset`` (``sequence``): the generic cached reader the SASRec dataset builds on.
"""

from collections import defaultdict
from irec.dataset.samplers import TrainSampler, EvalSampler
import pickle
import os
import logging
from .base import BaseDataset, BaseSequenceDataset

logger = logging.getLogger(__name__)


class SequenceDataset(BaseDataset, config_name='sequence'):
    """Generic reader of one sequence file per split with pickle caching; used by the SASRec dataset for its training file."""

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

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(
            config['path_to_data_dir'],
            config['name'],
        )

        common_params_for_creation = {
            'dir_path': data_dir_path,
            'max_sequence_length': config['max_sequence_length'],
            'use_cached': config.get('use_cached', False),
        }

        train_dataset, train_max_user_id, train_max_item_id, train_seq_len = cls._create_dataset(
            part='train', **common_params_for_creation
        )
        validation_dataset, valid_max_user_id, valid_max_item_id, valid_seq_len = cls._create_dataset(
            part='valid', **common_params_for_creation
        )
        test_dataset, test_max_user_id, test_max_item_id, test_seq_len = cls._create_dataset(
            part='test', **common_params_for_creation
        )

        max_user_id = max(
            [train_max_user_id, valid_max_user_id, test_max_user_id],
        )
        max_item_id = max(
            [train_max_item_id, valid_max_item_id, test_max_item_id],
        )
        max_seq_len = max([train_seq_len, valid_seq_len, test_seq_len])

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_seq_len))

        train_interactions = sum(
            list(map(lambda x: len(x), train_dataset)),
        )  # whole user history as a sample
        valid_interactions = len(
            validation_dataset,
        )  # each new interaction as a sample
        test_interactions = len(
            test_dataset,
        )  # each new interaction as a sample
        logger.info(
            '{} dataset sparsity: {}'.format(
                config['name'],
                (train_interactions + valid_interactions + test_interactions) / max_user_id / max_item_id,
            ),
        )

        samplers_config = config['samplers']
        train_sampler = TrainSampler.create_from_config(
            samplers_config,
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        validation_sampler = EvalSampler.create_from_config(
            samplers_config,
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        test_sampler = EvalSampler.create_from_config(
            samplers_config,
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_seq_len,
        )

    @classmethod
    def _create_dataset(
        cls,
        dir_path,
        part,
        max_sequence_length=None,
        use_cached=False,
    ):
        cache_path = os.path.join(dir_path, '{}.pkl'.format(part))

        if use_cached and os.path.exists(cache_path):
            logger.info('Loading cached dataset from {}'.format(cache_path))
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        return cls._build_and_cache_dataset(dir_path, part, max_sequence_length, cache_path, use_cached)

    @classmethod
    def _build_and_cache_dataset(cls, dir_path, part, max_sequence_length, cache_path, use_cached):
        logger.info('Cache is forcefully ignored.' if not use_cached else 'No cached dataset has been found.')
        dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
        logger.info('Creating a dataset from {}...'.format(dataset_path))

        with open(dataset_path, 'r') as f:
            data = f.readlines()

        sequence_info = cls._create_sequences(data, max_sequence_length)
        (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_len,
        ) = sequence_info

        dataset = []
        for user_id, item_ids in zip(user_sequences, item_sequences):
            dataset.append(
                {
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids,
                    'item.length': len(item_ids),
                },
            )

        logger.info('{} dataset size: {}'.format(part, len(dataset)))
        logger.info(
            '{} dataset max sequence length: {}'.format(
                part,
                max_sequence_len,
            ),
        )

        with open(cache_path, 'wb') as dataset_file:
            pickle.dump(
                (dataset, max_user_id, max_item_id, max_sequence_len),
                dataset_file,
            )

        return dataset, max_user_id, max_item_id, max_sequence_len

    @staticmethod
    def _create_sequences(data, max_sample_len):  # TODO
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]][-max_sample_len:]
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


class MCLSRDataset(BaseSequenceDataset, config_name='mclsr'):
    """The study's dataset: the prefix ladder for training, histories and targets for validation / test, items seen per user for negative sampling."""

    @staticmethod
    def _create_sequences_from_file(filepath, max_len=None):
        sequences = {}
        max_user, max_item = 0, 0

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]
                if max_len:
                    item_ids = item_ids[-max_len:]
                sequences[user_id] = item_ids
                max_user = max(max_user, user_id)
                if item_ids:
                    max_item = max(max_item, max(item_ids))
        return sequences, max_user, max_item

    @staticmethod
    def _read_train_samples(filepath, max_len=None):
        samples = []
        max_user, max_item = 0, 0

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]
                if max_len:
                    item_ids = item_ids[-max_len:]
                samples.append((user_id, item_ids))
                max_user = max(max_user, user_id)
                if item_ids:
                    max_item = max(max_item, max(item_ids))
        return samples, max_user, max_item

    @classmethod
    def _create_evaluation_sets(cls, data_dir, max_seq_len):
        # NB: train ladder lines are cut to max_seq_len and then lose their last
        # item to the target, so train INPUTS are <= max_seq_len - 1 items while
        # eval histories here get max_seq_len — the last position embedding is
        # never trained. Measured empirically zero on Clothing (0.4% of users
        # reach the cap, delta ±0.0001); set `eval_history_max_len` in the
        # dataset config to max_seq_len - 1 to align (pre-registered protocols).
        valid_hist, u2, i2 = cls._create_sequences_from_file(os.path.join(data_dir, 'valid_history.txt'), max_seq_len)
        valid_trg, u3, i3 = cls._create_sequences_from_file(os.path.join(data_dir, 'valid_target.txt'))

        validation_dataset = [
            {'user.ids': [uid], 'history': valid_hist[uid], 'target': valid_trg[uid]}
            for uid in valid_hist
            if uid in valid_trg
        ]

        test_hist, u4, i4 = cls._create_sequences_from_file(os.path.join(data_dir, 'test_history.txt'), max_seq_len)
        test_trg, u5, i5 = cls._create_sequences_from_file(os.path.join(data_dir, 'test_target.txt'))

        test_dataset = [
            {'user.ids': [uid], 'history': test_hist[uid], 'target': test_trg[uid]}
            for uid in test_hist
            if uid in test_trg
        ]

        return validation_dataset, test_dataset, max(u2, u3, u4, u5), max(i2, i3, i4, i5)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config['path_to_data_dir'], config['name'])
        max_seq_len = config.get('max_sequence_length')

        train_samples, u1, i1 = cls._read_train_samples(os.path.join(data_dir, 'train_mclsr.txt'), max_seq_len)
        train_dataset = [
            {'user.ids': [uid], 'user.length': 1, 'item.ids': seq, 'item.length': len(seq)}
            for uid, seq in train_samples
        ]

        user_to_all_seen_items = defaultdict(set)
        for sample in train_dataset:
            user_to_all_seen_items[sample['user.ids'][0]].update(sample['item.ids'])
        kwargs['user_to_all_seen_items'] = user_to_all_seen_items

        eval_hist_len = config.get('eval_history_max_len', max_seq_len)
        validation_dataset, test_dataset, u_eval, i_eval = cls._create_evaluation_sets(data_dir, eval_hist_len)
        num_users = max(u1, u_eval)
        num_items = max(i1, i_eval)

        train_sampler = TrainSampler.create_from_config(
            config['samplers'], dataset=train_dataset, num_users=num_users, num_items=num_items, **kwargs
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'], dataset=validation_dataset, num_users=num_users, num_items=num_items, **kwargs
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'], dataset=test_dataset, num_users=num_users, num_items=num_items, **kwargs
        )

        return cls(train_sampler, validation_sampler, test_sampler, num_users, num_items, max_seq_len)


class SASRecDataset(BaseSequenceDataset, config_name='sasrec_comparison'):
    """The SASRec comparison dataset: SASRec's training file (or the ladder) with exactly the MCLSR evaluation sets, so both encoders are scored on the same users, histories and targets."""

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config['path_to_data_dir'], config['name'])
        max_seq_len = config.get('max_sequence_length')

        # 'train_sasrec' = one full sequence per user (all positions are
        # queries); 'train_mclsr' = the prefix ladder, one line per query
        train_dataset, u1, i1, _ = SequenceDataset._create_dataset(
            dir_path=data_dir, part=config.get('train_part', 'train_sasrec'), max_sequence_length=max_seq_len
        )

        validation_dataset, test_dataset, u_eval, i_eval = MCLSRDataset._create_evaluation_sets(data_dir, max_seq_len)

        num_users = max(u1, u_eval)
        num_items = max(i1, i_eval)

        train_sampler = TrainSampler.create_from_config(
            config['train_sampler'], dataset=train_dataset, num_users=num_users, num_items=num_items
        )
        validation_sampler = EvalSampler.create_from_config(
            config['eval_sampler'], dataset=validation_dataset, num_users=num_users, num_items=num_items
        )
        test_sampler = EvalSampler.create_from_config(
            config['eval_sampler'], dataset=test_dataset, num_users=num_users, num_items=num_items
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=num_users,
            num_items=num_items,
            max_sequence_length=max_seq_len,
        )
