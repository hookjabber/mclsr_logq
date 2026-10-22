"""Sequence datasets of the study.

* ``MCLSRDataset`` (``mclsr``): the prefix ladder ``train_mclsr.txt`` (one training example per target
  position) and the validation / test histories and targets, as written by ``scripts/prepare_dataset.py``;
* ``SASRecDataset`` (``sasrec_comparison``): the same evaluation sets with ``train_sasrec.txt`` (one full
  cropped sequence per user, every position a query) or the ladder as the training file;
* ``SequenceDataset`` (``sequence``): the generic cached reader the SASRec dataset builds on.
"""

from collections import Counter, defaultdict
from irec.dataset.samplers import TrainSampler, EvalSampler
import pickle
import os
import logging
import json
import hashlib
from .base import BaseDataset, BaseSequenceDataset

logger = logging.getLogger(__name__)


def _sequence_lengths(config, data_dir):
    if os.path.exists(os.path.join(data_dir, '.preprocessing-in-progress')):
        raise ValueError('dataset preprocessing is incomplete; use a new completed dataset directory')
    manifest_path = os.path.join(data_dir, 'preprocessing.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise ValueError('preprocessing manifest must be an object')
        if manifest.get('protocol') != 'windowed-v2':
            raise ValueError('unsupported preprocessing protocol')
        if type(manifest.get('max_history_length')) is not int or manifest['max_history_length'] < 1:
            raise ValueError('preprocessing manifest requires a positive max_history_length')
        filenames = (
            'train_mclsr.txt',
            'train_sasrec.txt',
            'valid_history.txt',
            'valid_target.txt',
            'test_history.txt',
            'test_target.txt',
        )
        file_hashes = manifest.get('files_sha256')
        if not isinstance(file_hashes, dict) or set(file_hashes) != set(filenames):
            raise ValueError('preprocessing manifest must identify all six dataset file hashes')
    history_length = config.get('max_history_length')
    if history_length is None:
        history_length = manifest.get('max_history_length')
    if history_length is None:
        legacy_length = config.get('max_sequence_length')
        return legacy_length, config.get('eval_history_max_len', legacy_length), legacy_length, manifest
    if type(history_length) is not int or history_length < 1:
        raise ValueError('max_history_length must be a positive integer')
    for field, value in (
        ('manifest history length', manifest.get('max_history_length')),
        ('max_sequence_length', config.get('max_sequence_length')),
        ('eval history length', config.get('eval_history_max_len')),
    ):
        if value is not None and (type(value) is not int or value < 1 or value != history_length):
            raise ValueError(f'{field} conflicts with max_history_length={history_length}')
    if manifest:
        num_users, num_items = _catalogue_dimensions(manifest, 0, 0)
        _read_canonical_events(data_dir, manifest['files_sha256'], num_users, num_items, history_length)
    return history_length + 1, history_length, history_length, manifest


def _catalogue_dimensions(manifest, observed_users, observed_items):
    if not manifest:
        return observed_users, observed_items
    dimensions = []
    for field, observed in [('num_users', observed_users), ('num_items', observed_items)]:
        dimension = manifest.get(field)
        if type(dimension) is not int or dimension < 1 or dimension < observed:
            raise ValueError(f'preprocessing manifest catalogue {field} must cover all observed IDs')
        dimensions.append(dimension)
    return tuple(dimensions)


def _read_canonical_events(source_directory, source_hashes, num_users, num_items, history_length):
    raw_sequences = {}
    for filename, expected_hash in source_hashes.items():
        with open(os.path.join(source_directory, filename), 'rb') as source:
            payload = source.read()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f'canonical event input hash mismatch: {filename}')
        rows = []
        for line in payload.splitlines():
            try:
                user_id, *items = map(int, line.split())
            except ValueError as error:
                raise ValueError(f'invalid canonical event input: {filename}') from error
            if not 1 <= user_id <= num_users or not items or any(not 1 <= item_id <= num_items for item_id in items):
                raise ValueError(
                    f'canonical event input requires valid non-padding IDs within catalogue bounds: {filename}'
                )
            rows.append((user_id, tuple(items)))
        raw_sequences[filename] = rows
    raw_windows = Counter(raw_sequences['train_mclsr.txt'])
    remaining_windows = raw_windows.copy()
    heldout_users = set()
    for split in ('valid', 'test'):
        histories = raw_sequences[f'{split}_history.txt']
        targets = raw_sequences[f'{split}_target.txt']
        history_by_user = dict(histories)
        target_by_user = dict(targets)
        if (
            len(history_by_user) != len(histories)
            or len(target_by_user) != len(targets)
            or history_by_user.keys() != target_by_user.keys()
            or heldout_users.intersection(history_by_user)
        ):
            raise ValueError('canonical evaluation histories/targets require matching disjoint unique users')
        for user_id, history in histories:
            total_length = len(history) + len(target_by_user[user_id])
            if total_length < 5 or len(history) != 4 * total_length // 5:
                raise ValueError('canonical evaluation requires the raw 80/20 history/target split')
        heldout_users.update(history_by_user)
    seen_users = set()
    sequences = []
    for user_id, items in raw_sequences['train_sasrec.txt']:
        if user_id in seen_users or user_id in heldout_users or len(items) < 2:
            raise ValueError('canonical event sequences require unique training users and valid non-padding IDs')
        seen_users.add(user_id)
        for target_index in range(1, len(items)):
            start = max(0, target_index - history_length)
            window = (user_id, tuple(items[start : target_index + 1]))
            if remaining_windows[window] == 0:
                raise ValueError('canonical events do not match training windows')
            remaining_windows[window] -= 1
            if remaining_windows[window] == 0:
                del remaining_windows[window]
        sequences.append((user_id, items))
    if remaining_windows:
        raise ValueError('training windows are not covered by canonical events')
    return sequences, raw_windows


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


class MCLSRDataset(BaseSequenceDataset, config_name='mclsr'):
    """The study's dataset: the prefix ladder for training, histories and targets for validation / test, items seen per user for negative sampling."""

    def get_train_event_sequences(self):
        """Return canonical v2 events after checking their hash and exact window multiset.

        Repeated items are real events here. Duplicate ladder rows can be legitimate,
        but must have exactly the multiplicity generated by the canonical sequence.
        Legacy/cropped inputs cannot establish this contract and are rejected.
        """
        source_info = getattr(self, '_train_event_source', None)
        if source_info is None:
            raise ValueError('canonical event counts require a windowed-v2 MCLSR dataset')
        source_directory, source_hashes = source_info
        sequences, raw_windows = _read_canonical_events(
            source_directory, source_hashes, self.num_users, self.num_items, self.max_sequence_length
        )
        loaded_windows = Counter((row['user.ids'][0], tuple(row['item.ids'])) for row in self._train_sampler.dataset)
        if raw_windows != loaded_windows:
            raise ValueError('raw training windows have been cropped or changed by the loader')
        return sequences

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
        train_row_len, eval_hist_len, max_seq_len, manifest = _sequence_lengths(config, data_dir)

        train_samples, u1, i1 = cls._read_train_samples(os.path.join(data_dir, 'train_mclsr.txt'), train_row_len)
        train_dataset = [
            {'user.ids': [uid], 'user.length': 1, 'item.ids': seq, 'item.length': len(seq)}
            for uid, seq in train_samples
        ]

        user_to_all_seen_items = defaultdict(set)
        for sample in train_dataset:
            user_to_all_seen_items[sample['user.ids'][0]].update(sample['item.ids'])
        kwargs['user_to_all_seen_items'] = user_to_all_seen_items

        validation_dataset, test_dataset, u_eval, i_eval = cls._create_evaluation_sets(data_dir, eval_hist_len)
        num_users, num_items = _catalogue_dimensions(manifest, max(u1, u_eval), max(i1, i_eval))

        train_sampler = TrainSampler.create_from_config(
            config['samplers'], dataset=train_dataset, num_users=num_users, num_items=num_items, **kwargs
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'], dataset=validation_dataset, num_users=num_users, num_items=num_items, **kwargs
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'], dataset=test_dataset, num_users=num_users, num_items=num_items, **kwargs
        )

        dataset = cls(train_sampler, validation_sampler, test_sampler, num_users, num_items, max_seq_len)
        dataset._train_event_source = (data_dir, manifest['files_sha256']) if manifest else None
        return dataset


class SASRecDataset(BaseSequenceDataset, config_name='sasrec_comparison'):
    """The SASRec comparison dataset: SASRec's training file (or the ladder) with exactly the MCLSR evaluation sets, so both encoders are scored on the same users, histories and targets."""

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config['path_to_data_dir'], config['name'])
        train_row_len, eval_hist_len, max_seq_len, manifest = _sequence_lengths(config, data_dir)
        train_part = config.get('train_part', 'train_sasrec')
        if manifest and train_part not in ('train_sasrec', 'train_mclsr'):
            raise ValueError('windowed-v2 train_part must select a canonical train_sasrec or train_mclsr source')

        # 'train_sasrec' = one full sequence per user (all positions are
        # queries); 'train_mclsr' = the prefix ladder, one line per query
        train_dataset, u1, i1, _ = SequenceDataset._create_dataset(
            dir_path=data_dir, part=train_part, max_sequence_length=train_row_len
        )

        validation_dataset, test_dataset, u_eval, i_eval = MCLSRDataset._create_evaluation_sets(data_dir, eval_hist_len)

        num_users, num_items = _catalogue_dimensions(manifest, max(u1, u_eval), max(i1, i_eval))

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
