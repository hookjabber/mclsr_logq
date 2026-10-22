import math
import pickle
from types import SimpleNamespace

import pytest
import torch

import irec.utils
from irec.callbacks.base import EvalCallback, ValidationCallback
from irec.dataset.sequences import SequenceDataset
from irec.metric.base import MCLSRHitRateMetric, MCLSRNDCGMetric, MCLSRRecallMetric


@pytest.mark.parametrize('requested_length', [None, 2, 4])
def test_sequence_cache_respects_current_history_capacity(tmp_path, requested_length):
    (tmp_path / 'train.txt').write_text('1 1 2 3 4\n')
    SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True)

    actual = SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=requested_length, use_cached=True)

    expected_items = [1, 2, 3, 4] if requested_length is None else [1, 2, 3, 4][-requested_length:]
    assert actual[0][0]['item.ids'] == expected_items
    assert actual[3] == len(expected_items)


def test_sequence_cache_respects_changed_source_events_and_catalogue(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text('1 1 2 3\n')
    SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True)
    source.write_text('2 4 4 5\n')

    actual = SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True)

    assert actual[0] == [{'user.ids': [2], 'user.length': 1, 'item.ids': [4, 4, 5], 'item.length': 3}]
    assert actual[1:] == (2, 5, 3)


def test_sequence_cache_without_source_identity_is_rebuilt(tmp_path):
    (tmp_path / 'train.txt').write_text('2 4 4 5\n')
    stale = ([{'user.ids': [1], 'user.length': 1, 'item.ids': [1, 2], 'item.length': 2}], 1, 2, 2)
    with (tmp_path / 'train.pkl').open('wb') as destination:
        pickle.dump(stale, destination)

    actual = SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True)

    assert actual[0][0]['item.ids'] == [4, 4, 5]
    assert actual[1:] == (2, 5, 3)


def test_unchanged_sequence_cache_is_reused(tmp_path, monkeypatch):
    (tmp_path / 'train.txt').write_text('1 1 2 3\n')
    expected = SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True)

    def fail_on_rebuild(*arguments, **kwargs):
        pytest.fail('an unchanged source and capacity should reuse the cache')

    monkeypatch.setattr(SequenceDataset, '_build_and_cache_dataset', fail_on_rebuild)
    assert SequenceDataset._create_dataset(tmp_path, 'train', max_sequence_length=3, use_cached=True) == expected


@pytest.mark.parametrize('callback_class', [ValidationCallback, EvalCallback])
def test_empty_evaluation_cannot_publish_nan_selection_metric(callback_class, monkeypatch):
    recorded = []
    writer = SimpleNamespace(add_scalar=lambda *values: recorded.append(values), flush=lambda: None)
    monkeypatch.setattr(irec.utils.tensorboards, 'GLOBAL_TENSORBOARD_WRITER', writer)
    callback = callback_class(
        model=torch.nn.Identity(),
        train_dataloader=None,
        validation_dataloader=[],
        eval_dataloader=[],
        optimizer=None,
        on_step=1,
        pred_prefix='predictions',
        labels_prefix='labels',
        metrics={'recall@1': MCLSRRecallMetric(1)},
    )
    output = {}

    with pytest.raises(ValueError, match='evaluation.*empty|evaluation.*no.*(user|metric|value)'):
        callback(output, 0)

    assert output == {}
    assert recorded == []


class _FixedRanking(torch.nn.Module):
    def forward(self, inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        return inputs['ranking']


@pytest.mark.parametrize('callback_class', [ValidationCallback, EvalCallback])
@pytest.mark.parametrize('partition', [(3,), (2, 1), (1, 1, 1)])
def test_callback_macro_average_preserves_set_relevance_and_empty_target_users(callback_class, partition, monkeypatch):
    monkeypatch.setattr(irec.utils, 'DEVICE', torch.device('cpu'))
    writer = SimpleNamespace(add_scalar=lambda *values: None, flush=lambda: None)
    monkeypatch.setattr(irec.utils.tensorboards, 'GLOBAL_TENSORBOARD_WRITER', writer)
    predictions = [[1, 2], [3, 4], [4, 3]]
    targets = [[1, 1, 5], [3], []]
    batches = []
    offset = 0
    for size in partition:
        batch_targets = targets[offset : offset + size]
        batches.append(
            {
                'ranking': torch.tensor(predictions[offset : offset + size]),
                'labels.ids': torch.tensor([item for row in batch_targets for item in row], dtype=torch.long),
                'labels.length': torch.tensor([len(row) for row in batch_targets]),
            }
        )
        offset += size
    callback = callback_class(
        model=_FixedRanking(),
        train_dataloader=None,
        validation_dataloader=batches,
        eval_dataloader=batches,
        optimizer=None,
        on_step=1,
        pred_prefix='predictions',
        labels_prefix='labels',
        metrics={'recall': MCLSRRecallMetric(2), 'ndcg': MCLSRNDCGMetric(2), 'hit': MCLSRHitRateMetric(2)},
    )

    output = {}
    callback(output, 0)

    expected = {'recall': 0.5, 'ndcg': (1 / (1 + 1 / math.log2(3)) + 1) / 3, 'hit': 2 / 3}
    assert output == pytest.approx({f'{callback.config_name}/{name}': value for name, value in expected.items()})
