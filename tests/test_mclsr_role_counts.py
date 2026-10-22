import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from irec.dataset.sequences import MCLSRDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_dataset import write_files


def _run_counts(input_path, output_dir, max_len=None):
    target_path = output_dir / 'target_counts.pkl'
    context_path = output_dir / 'context_counts.pkl'
    command = [
        sys.executable,
        str(ROOT / 'scripts' / 'generate_mclsr_role_counts.py'),
        '--input',
        str(input_path),
        '--num_items',
        '25',
        '--target_output',
        str(target_path),
        '--context_output',
        str(context_path),
    ]
    if max_len is not None:
        command.extend(['--max_len', str(max_len)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return result, target_path, context_path


def _write_v2(tmp_path, history_length=20):
    directory = tmp_path / 'Tiny'
    sequences = [(0, list(range(25))), (8, list(range(25))), (30, list(range(25)))]
    write_files(sequences, {0}, {8}, {30}, directory, protocol='windowed-v2', max_history_length=history_length)
    return directory


@pytest.mark.parametrize('history_length', [1, 20, 30])
def test_v2_role_counts_match_actual_mclsr_sampler(tmp_path, history_length):
    directory = _write_v2(tmp_path, history_length)
    dataset = MCLSRDataset.create_from_config(
        {
            'path_to_data_dir': str(tmp_path),
            'name': 'Tiny',
            'max_sequence_length': history_length,
            'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
        }
    )
    train, _, _ = dataset.get_samplers()
    expected_target = np.zeros(dataset.num_items + 2, dtype=np.float32)
    expected_context = np.zeros(dataset.num_items + 2, dtype=np.float32)
    for index in range(len(train)):
        sample = train[index]
        for target in sample['labels.ids']:
            expected_target[target] += 1
        for item in set(sample['item.ids']):
            expected_context[item] += 1

    result, target_path, context_path = _run_counts(directory / 'train_mclsr.txt', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    with target_path.open('rb') as source:
        target_artifact = pickle.load(source)
    with context_path.open('rb') as source:
        context_artifact = pickle.load(source)
    np.testing.assert_array_equal(target_artifact['counts'], expected_target)
    np.testing.assert_array_equal(context_artifact['counts'], expected_context)
    assert len(train) == 24
    assert target_artifact['role'] == 'target'
    assert context_artifact['role'] == 'context-inclusion'
    for artifact in (target_artifact, context_artifact):
        assert artifact['max_len'] == history_length + 1
        assert artifact['denominator'] == 24
        assert artifact['num_lines'] == 24


@pytest.mark.parametrize('max_len', [None, 21])
def test_legacy_role_counts_preserve_default_and_explicit_lengths(tmp_path, max_len):
    input_path = tmp_path / 'train_mclsr.txt'
    input_path.write_text('1 ' + ' '.join(map(str, range(1, 26))) + '\n')
    result, target_path, context_path = _run_counts(input_path, tmp_path, max_len)
    assert result.returncode == 0, result.stdout + result.stderr
    with target_path.open('rb') as source:
        target_artifact = pickle.load(source)
    with context_path.open('rb') as source:
        context_artifact = pickle.load(source)
    expected_length = 20 if max_len is None else max_len
    expected_context = np.zeros(27, dtype=np.float32)
    expected_context[26 - expected_length : 25] = 1
    expected_target = np.zeros(27, dtype=np.float32)
    expected_target[25] = 1
    np.testing.assert_array_equal(context_artifact['counts'], expected_context)
    np.testing.assert_array_equal(target_artifact['counts'], expected_target)
    assert context_artifact['max_len'] == target_artifact['max_len'] == expected_length


def test_v2_role_counts_allow_matching_explicit_length(tmp_path):
    directory = _write_v2(tmp_path)
    result, target_path, context_path = _run_counts(directory / 'train_mclsr.txt', tmp_path, 21)
    assert result.returncode == 0, result.stdout + result.stderr
    with context_path.open('rb') as source:
        artifact = pickle.load(source)
    assert artifact['max_len'] == 21
    assert artifact['counts'][1] == 20
    assert target_path.exists()


@pytest.mark.parametrize('max_len', [19, 20, 22])
def test_v2_role_counts_reject_conflicting_explicit_length(tmp_path, max_len):
    directory = _write_v2(tmp_path)
    result, target_path, context_path = _run_counts(directory / 'train_mclsr.txt', tmp_path, max_len)
    assert result.returncode != 0
    assert 'max_len' in result.stderr and '21' in result.stderr
    assert not target_path.exists()
    assert not context_path.exists()


@pytest.mark.parametrize(
    'manifest',
    [
        {'protocol': 'unknown', 'max_history_length': 20},
        {'protocol': 'windowed-v2'},
        {'protocol': 'windowed-v2', 'max_history_length': None},
        {'protocol': 'windowed-v2', 'max_history_length': 0},
        {'protocol': 'windowed-v2', 'max_history_length': True},
        {'protocol': 'windowed-v2', 'max_history_length': 2.5},
        [],
    ],
)
def test_role_counts_reject_invalid_manifest(tmp_path, manifest):
    directory = _write_v2(tmp_path)
    (directory / 'preprocessing.json').write_text(json.dumps(manifest))
    result, target_path, context_path = _run_counts(directory / 'train_mclsr.txt', tmp_path)
    assert result.returncode != 0
    assert 'manifest' in result.stderr
    assert not target_path.exists()
    assert not context_path.exists()


@pytest.mark.parametrize('max_len', [0, -1])
def test_role_counts_reject_nonpositive_explicit_length(tmp_path, max_len):
    input_path = tmp_path / 'train_mclsr.txt'
    input_path.write_text('1 1 2 3\n')
    result, target_path, context_path = _run_counts(input_path, tmp_path, max_len)
    assert result.returncode != 0
    assert 'max_len' in result.stderr
    assert not target_path.exists()
    assert not context_path.exists()


def test_role_counts_reject_incomplete_preprocessing(tmp_path):
    directory = _write_v2(tmp_path)
    (directory / 'preprocessing.json').unlink()
    (directory / '.preprocessing-in-progress').touch()
    result, target_path, context_path = _run_counts(directory / 'train_mclsr.txt', tmp_path)
    assert result.returncode != 0
    assert 'incomplete' in result.stderr
    assert not target_path.exists()
    assert not context_path.exists()
