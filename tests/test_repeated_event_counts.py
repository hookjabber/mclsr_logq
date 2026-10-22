"""Repeated-event protocol oracles, without graphs, models, losses or external evidence."""

import hashlib
import importlib.util
import pickle
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import irec.utils
from irec.dataloader.batch_processors import BasicBatchProcessor
from irec.dataset.sequences import MCLSRDataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def scripts():
    modules = {}
    for name in ['prepare_dataset', 'generate_mclsr_role_counts']:
        spec = importlib.util.spec_from_file_location(f'repeated_event_{name}', ROOT / 'scripts' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(prepare=modules['prepare_dataset'], counts=modules['generate_mclsr_role_counts'])


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(irec.utils, 'DEVICE', torch.device('cpu'))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _event_position_oracle(sequences, train_users, history_length):
    item_map = {item: rank + 1 for rank, item in enumerate(sorted({item for _, items in sequences for item in items}))}
    user_map = {user: rank + 1 for rank, user in enumerate(sorted(user for user, _ in sequences))}
    rows = []
    target_counts = Counter()
    context_counts = Counter()
    for user, events in sequences:
        if user not in train_users:
            continue
        for target_position, target in enumerate(events):
            if target_position == 0:
                continue
            history = [
                item_map[item]
                for position, item in enumerate(events)
                if 0 < target_position - position <= history_length
            ]
            rows.append((user_map[user], history, item_map[target]))
            target_counts[item_map[target]] += 1
            for item, mapped_item in item_map.items():
                if any(
                    earlier_item == item and 0 < target_position - position <= history_length
                    for position, earlier_item in enumerate(events)
                ):
                    context_counts[mapped_item] += 1
    return SimpleNamespace(rows=rows, target=target_counts, context=context_counts, items=item_map, users=user_map)


def _generate_counts(input_path, num_items, scripts, monkeypatch, max_len=None):
    target_path = input_path.parent / 'target_counts.pkl'
    context_path = input_path.parent / 'context_counts.pkl'
    arguments = [
        'generate_mclsr_role_counts.py',
        '--input',
        str(input_path),
        '--num_items',
        str(num_items),
        '--target_output',
        str(target_path),
        '--context_output',
        str(context_path),
    ]
    if max_len is not None:
        arguments.extend(['--max_len', str(max_len)])
    with monkeypatch.context() as patch:
        patch.setattr(sys, 'argv', arguments)
        scripts.counts.main()
    with target_path.open('rb') as stream:
        target = pickle.load(stream)
    with context_path.open('rb') as stream:
        context = pickle.load(stream)
    return target, context


def _assert_counts(artifacts, target, context, num_items, rows, row_cap, input_path):
    for artifact, expected, role in zip(artifacts, [target, context], ['target', 'context-inclusion']):
        assert artifact['counts'].tolist() == [expected[item] for item in range(num_items + 2)]
        assert artifact['role'] == role
        assert artifact['num_lines'] == artifact['denominator'] == rows
        assert artifact['max_len'] == row_cap
        assert artifact['source_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
        assert artifact['counts'][0] == artifact['counts'][num_items + 1] == 0
        assert all(0 <= count <= rows for count in artifact['counts'])
    assert sum(target.values()) == rows
    assert float((artifacts[0]['counts'] / artifacts[0]['denominator']).sum()) == pytest.approx(1.0)


def _prepare_and_check(directory, sequences, history_length, scripts, monkeypatch, splits=None):
    train_users, valid_users, test_users = splits or ({user for user, _ in sequences}, set(), set())
    expected = _event_position_oracle(sequences, train_users, history_length)
    row_count = scripts.prepare.write_files(
        sequences,
        train_users,
        valid_users,
        test_users,
        directory,
        protocol='windowed-v2',
        max_history_length=history_length,
    )
    assert row_count == sum(len(items) - 1 for user, items in sequences if user in train_users)
    ladder_path = directory / 'train_mclsr.txt'
    serialized = [[int(value) for value in line.split()] for line in ladder_path.read_text().splitlines()]
    assert serialized == [[user, *history, target] for user, history, target in expected.rows]
    sequence_rows = [
        [int(value) for value in line.split()] for line in (directory / 'train_sasrec.txt').read_text().splitlines()
    ]
    assert sequence_rows == [
        [expected.users[user], *[expected.items[item] for item in items]]
        for user, items in sequences
        if user in train_users
    ]
    dataset = MCLSRDataset.create_from_config(
        {
            'path_to_data_dir': str(directory.parent),
            'name': directory.name,
            'max_history_length': history_length,
            'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
        }
    )
    train, validation, test = dataset.get_samplers()
    observed = []
    restored = defaultdict(list)
    for sample_index in range(len(train)):
        sample = train[sample_index]
        user = sample['user.ids'][0]
        history = sample['item.ids']
        target = sample['labels.ids'][0]
        assert sample['item.length'] == len(history)
        assert 1 <= len(history) <= history_length
        assert sample['labels.length'] == 1
        assert sample['negatives.ids'] == []
        assert all(item > 0 for item in [*history, target])
        observed.append((user, history, target))
        if not restored[user]:
            restored[user].append(history[0])
        restored[user].append(target)
    assert observed == expected.rows
    assert dict(restored) == {
        expected.users[user]: [expected.items[item] for item in items]
        for user, items in sequences
        if user in train_users
    }
    artifacts = _generate_counts(ladder_path, dataset.num_items, scripts, monkeypatch)
    _assert_counts(
        artifacts, expected.target, expected.context, dataset.num_items, row_count, history_length + 1, ladder_path
    )
    return SimpleNamespace(
        expected=expected, train=train, validation=validation, test=test, artifacts=artifacts, dataset=dataset
    )


def test_aba_return_visit_is_a_new_target_even_when_already_in_context(tmp_path, scripts, monkeypatch):
    result = _prepare_and_check(tmp_path / 'aba', [(9, [0, 7, 0])], 20, scripts, monkeypatch)
    assert result.expected.rows == [(1, [1], 2), (1, [1, 2], 1)]
    target, context = result.artifacts
    assert target['counts'].tolist() == [0, 1, 1, 0]
    assert context['counts'].tolist() == [0, 2, 1, 0]
    assert target['denominator'] == context['denominator'] == 2
    assert float((context['counts'] / context['denominator']).sum()) == 1.5


@pytest.mark.parametrize('history_length,expected_context', [(1, [3, 2]), (2, [4, 4]), (3, [4, 4]), (20, [4, 4])])
def test_context_deduplication_is_per_window_not_per_event_or_user(
    tmp_path, scripts, monkeypatch, history_length, expected_context
):
    sequences = [(40, [0, 7, 0, 0]), (90, [7, 0, 7])]
    result = _prepare_and_check(tmp_path / 'repeats', sequences, history_length, scripts, monkeypatch)
    target, context = result.artifacts
    assert target['counts'].tolist() == [0, 3, 2, 0]
    assert context['counts'].tolist() == [0, *expected_context, 0]
    assert len(result.train) == 5
    assert result.train[2]['labels.ids'][0] in result.train[2]['item.ids']
    if history_length >= 3:
        assert result.train[2]['item.ids'] == [1, 2, 1]
        event_frequency = Counter(item for _, history, _ in result.expected.rows for item in history)
        assert event_frequency[1] == 5
        assert context['counts'][1] == 4


@pytest.mark.parametrize('history_length', [1, 2, 3, 20])
def test_finite_binary_sequence_enumeration_preserves_events_windows_and_counts(
    tmp_path, scripts, monkeypatch, history_length
):
    enumerated = [list(events) for length in range(2, 7) for events in product([0, 7], repeat=length)]
    sequences = [(10 * index, items) for index, items in enumerate(enumerated)]
    assert len(sequences) == 124
    result = _prepare_and_check(tmp_path / 'enumeration', sequences, history_length, scripts, monkeypatch)
    assert len(result.train) == 516
    for batch_size in [1, 7, 129]:
        reconstructed_target = Counter()
        reconstructed_context = Counter()
        padded_slots = 0
        for start in range(0, len(result.train), batch_size):
            samples = [result.train[index] for index in range(start, min(start + batch_size, len(result.train)))]
            batch = BasicBatchProcessor()(samples)
            padded, mask = irec.utils.create_masked_tensor(batch['item.ids'], batch['item.length'])
            assert padded.device.type == mask.device.type == 'cpu'
            assert padded[mask].tolist() == [item for sample in samples for item in sample['item.ids']]
            assert all(value == 0 for value in padded[~mask].tolist())
            assert all(value > 0 for value in padded[mask].tolist())
            padded_slots += int((~mask).sum())
            for row, sample in enumerate(samples):
                actual_history = padded[row][mask[row]].tolist()
                assert actual_history == sample['item.ids']
                reconstructed_context.update(set(actual_history))
                reconstructed_target.update(sample['labels.ids'])
        assert reconstructed_target == result.expected.target
        assert reconstructed_context == result.expected.context
        assert reconstructed_context[0] == reconstructed_target[0] == 0
        if history_length > 1 and batch_size > 1:
            assert padded_slots > 0


@pytest.mark.parametrize('history_length', [1, 3, 20])
def test_raw_and_sequence_inputs_roundtrip_repeated_events_into_identical_v2_windows(
    tmp_path, scripts, monkeypatch, history_length
):
    raw_sequences = [
        ('user-a', ['A', 'B', 'A'] * 9 + ['A']),
        ('user-b', ['B', 'A', 'B'] * 2),
        ('user-c', ['A', 'B'] * 4 + ['A', 'A']),
        ('user-d', ['C'] * 5),
    ]
    events = [
        (user, item, 1, 1000 + position if not (user == 'user-a' and position == 27) else 1026)
        for user, items in raw_sequences
        for position, item in enumerate(items)
    ]
    assert Counter(events)[('user-a', 'A', 1, 1026)] == 2
    raw_path = tmp_path / 'raw.csv'
    raw_path.write_text(''.join(','.join(map(str, event)) + '\n' for event in reversed(events)))
    item_map = {'A': 1, 'B': 2, 'C': 3}
    expected_sequences = [
        (rank, [item_map[item] for item in items]) for rank, (_, items) in enumerate(raw_sequences, 1)
    ]
    from_csv = scripts.prepare.sequences_from_csv(raw_path)
    assert from_csv == expected_sequences
    seq_path = tmp_path / 'source.seq'
    seq_path.write_text(''.join(' '.join(map(str, [user, *items])) + '\n' for user, items in expected_sequences))
    from_seq = scripts.prepare.sequences_from_seq_file(seq_path)
    assert from_seq == expected_sequences
    splits = ({1, 2}, {3}, {4})
    csv_result = _prepare_and_check(tmp_path / 'from_csv', from_csv, history_length, scripts, monkeypatch, splits)
    seq_result = _prepare_and_check(tmp_path / 'from_seq', from_seq, history_length, scripts, monkeypatch, splits)
    assert len(csv_result.train) == len(seq_result.train) == 32
    assert csv_result.expected.rows == seq_result.expected.rows
    assert (tmp_path / 'from_csv/train_mclsr.txt').read_bytes() == (tmp_path / 'from_seq/train_mclsr.txt').read_bytes()
    assert scripts.prepare.sequences_from_seq_file(tmp_path / 'from_csv/train_sasrec.txt') == expected_sequences[:2]
    assert csv_result.validation[0]['labels.ids'] == [1, 1]
    assert csv_result.test[0]['labels.ids'] == [3]
    assert csv_result.artifacts[0]['counts'][3] == csv_result.artifacts[1]['counts'][3] == 0


@pytest.mark.parametrize('history_length', [1, 2, 20])
def test_repeated_heldout_targets_and_overlap_are_preserved_without_entering_train_counts(
    tmp_path, scripts, monkeypatch, history_length
):
    sequences = [
        (11, [0, 7, 0, 0, 7]),
        (22, [0, 7, 0, 7, 0, 7, 7, 0, 0, 0]),
        (33, [99, 7, 99, 7, 99, 7, 7, 99, 99, 99]),
    ]
    result = _prepare_and_check(
        tmp_path / 'heldout', sequences, history_length, scripts, monkeypatch, ({11}, {22}, {33})
    )
    for sampler, original in [(result.validation, sequences[1][1]), (result.test, sequences[2][1])]:
        boundary = 4 * len(original) // 5
        expected_history = [
            result.expected.items[item]
            for position, item in enumerate(original)
            if 0 < boundary - position <= history_length
        ]
        expected_targets = [
            result.expected.items[item] for position, item in enumerate(original) if position >= boundary
        ]
        assert sampler[0]['item.ids'] == expected_history
        assert sampler[0]['labels.ids'] == expected_targets
        assert sampler[0]['labels.length'] == 2
        assert expected_targets[0] == expected_targets[1]
        assert expected_targets[0] in expected_history
    target, context = result.artifacts
    assert target['counts'].tolist() == [0, 2, 2, 0, 0]
    assert target['denominator'] == context['denominator'] == 4
    assert context['counts'][result.expected.items[99]] == 0


@pytest.mark.parametrize('row_cap', [2, 3, 21])
def test_count_generator_caps_before_context_presence_and_does_not_remove_overlapping_target(
    tmp_path, scripts, monkeypatch, row_cap
):
    rows = [(19, [1, 2, 2, 2, 1]), (19, [1, 1, 1, 1]), (73, [2, 1, 2] * 8)]
    input_path = tmp_path / 'train_mclsr.txt'
    input_path.write_text(''.join(' '.join(map(str, [user, *items])) + '\n' for user, items in rows))
    target = Counter(items[-1] for _, items in rows)
    context = Counter()
    for candidate in [1, 2, 3]:
        context[candidate] = sum(
            any(item == candidate and 0 < len(items) - 1 - position < row_cap for position, item in enumerate(items))
            for _, items in rows
        )
    artifacts = _generate_counts(input_path, 3, scripts, monkeypatch, max_len=row_cap)
    _assert_counts(artifacts, target, context, 3, 3, row_cap, input_path)
    assert artifacts[0]['counts'].tolist() == [0, 2, 1, 0, 0]
    assert artifacts[1]['counts'].tolist() == [0, 2 if row_cap < 5 else 3, 1 if row_cap == 2 else 2, 0, 0]


def test_repeating_a_user_sequence_changes_counts_linearly_not_by_global_deduplication(tmp_path, scripts, monkeypatch):
    base = [(10, [0, 7, 0, 0, 7, 0])]
    doubled = base + [(90, list(base[0][1]))]
    single = _prepare_and_check(tmp_path / 'single', base, 3, scripts, monkeypatch)
    repeated = _prepare_and_check(tmp_path / 'doubled', doubled, 3, scripts, monkeypatch)
    reordered = _prepare_and_check(tmp_path / 'reordered', list(reversed(doubled)), 3, scripts, monkeypatch)
    for single_artifact, repeated_artifact, reordered_artifact in zip(
        single.artifacts, repeated.artifacts, reordered.artifacts
    ):
        assert repeated_artifact['counts'].tolist() == (2 * single_artifact['counts']).tolist()
        assert repeated_artifact['denominator'] == 2 * single_artifact['denominator']
        assert repeated_artifact['counts'].tolist() == reordered_artifact['counts'].tolist()
        assert (repeated_artifact['counts'] / repeated_artifact['denominator']).tolist() == pytest.approx(
            (single_artifact['counts'] / single_artifact['denominator']).tolist()
        )
