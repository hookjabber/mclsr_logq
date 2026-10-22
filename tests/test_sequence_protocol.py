import json
import hashlib

import pytest
import torch

from irec.dataset.sequences import MCLSRDataset, SASRecDataset
from irec.dataloader.batch_processors import BasicBatchProcessor
from irec.models import BaseModel


def _dataset_config(tmp_path, kind='mclsr'):
    directory = tmp_path / 'Tiny'
    directory.mkdir()
    train_items = list(range(1, 26))
    windows = [train_items[max(0, target_index - 20) : target_index + 1] for target_index in range(1, len(train_items))]
    windows.sort(key=len, reverse=True)
    (directory / 'train_mclsr.txt').write_text(''.join('1 ' + ' '.join(map(str, window)) + '\n' for window in windows))
    (directory / 'train_sasrec.txt').write_text('1 ' + ' '.join(map(str, range(1, 26))) + '\n')
    evaluation_items = list(range(1, 26)) + list(range(1, 6))
    split_index = 4 * len(evaluation_items) // 5
    for split, user in [('valid', 2), ('test', 3)]:
        (directory / f'{split}_history.txt').write_text(
            f'{user} ' + ' '.join(map(str, evaluation_items[:split_index])) + '\n'
        )
        (directory / f'{split}_target.txt').write_text(
            f'{user} ' + ' '.join(map(str, evaluation_items[split_index:])) + '\n'
        )
    (directory / 'preprocessing.json').write_text(
        json.dumps(
            {
                'protocol': 'windowed-v2',
                'max_history_length': 20,
                'num_items': 25,
                'num_users': 3,
                'files_sha256': {
                    file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in directory.glob('*.txt')
                },
            }
        )
    )
    return {
        'type': kind,
        'path_to_data_dir': str(tmp_path),
        'name': 'Tiny',
        'max_sequence_length': 20,
        'samplers': {'type': 'mclsr', 'num_negatives_train': 0},
        'train_sampler': {'type': 'next_item_prediction', 'negative_sampler_type': 'random', 'num_negatives_train': 0},
        'eval_sampler': {'type': 'mclsr'},
    }


@pytest.mark.parametrize('dataset_class', [MCLSRDataset, SASRecDataset])
def test_v2_training_and_eval_use_same_history_capacity(tmp_path, dataset_class):
    config = _dataset_config(tmp_path)
    dataset = dataset_class.create_from_config(config)
    train, validation, test = dataset.get_samplers()
    assert train[0]['item.length'] == 20
    assert validation[0]['item.length'] == test[0]['item.length'] == 20
    assert dataset.max_sequence_length == 20


def test_v2_manifest_rejects_conflicting_history_length(tmp_path):
    config = _dataset_config(tmp_path)
    config['max_history_length'] = 19
    with pytest.raises(ValueError, match='history'):
        MCLSRDataset.create_from_config(config)


def test_legacy_loading_is_unchanged(tmp_path):
    config = _dataset_config(tmp_path)
    (tmp_path / 'Tiny' / 'preprocessing.json').unlink()
    dataset = MCLSRDataset.create_from_config(config)
    train, validation, _ = dataset.get_samplers()
    assert train[0]['item.length'] == 19
    assert validation[0]['item.length'] == 20


@pytest.mark.parametrize('dataset_class', [MCLSRDataset, SASRecDataset])
def test_legacy_catalogue_size_retains_checkpoint_shape(tmp_path, dataset_class):
    config = _dataset_config(tmp_path)
    for filename in ['train_mclsr.txt', 'train_sasrec.txt']:
        (tmp_path / 'Tiny' / filename).write_text('1 90 ' + ' '.join(map(str, range(1, 25))) + '\n')
    (tmp_path / 'Tiny' / 'preprocessing.json').unlink()
    dataset = dataset_class.create_from_config(config)
    assert dataset.num_items == 25


@pytest.mark.parametrize('dataset_class', [MCLSRDataset, SASRecDataset])
def test_v2_catalogue_uses_full_manifest_even_for_cropped_training(tmp_path, dataset_class):
    config = _dataset_config(tmp_path)
    manifest_path = tmp_path / 'Tiny' / 'preprocessing.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(num_items=90, num_users=8)
    manifest_path.write_text(json.dumps(manifest))
    dataset = dataset_class.create_from_config(config)
    assert dataset.num_items == 90
    assert dataset.num_users == 8


@pytest.mark.parametrize('dataset_class', [MCLSRDataset, SASRecDataset])
def test_v2_null_history_setting_inherits_manifest(tmp_path, dataset_class):
    config = _dataset_config(tmp_path)
    config['max_history_length'] = None
    dataset = dataset_class.create_from_config(config)
    train, validation, test = dataset.get_samplers()
    assert train[0]['item.length'] == validation[0]['item.length'] == test[0]['item.length'] == 20


@pytest.mark.parametrize('field,value', [('num_items', None), ('num_users', 0), ('num_items', 24)])
def test_v2_rejects_invalid_catalogue_dimensions(tmp_path, field, value):
    config = _dataset_config(tmp_path)
    manifest_path = tmp_path / 'Tiny' / 'preprocessing.json'
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='catalogue'):
        MCLSRDataset.create_from_config(config)


def test_v2_longest_position_receives_training_gradient(tmp_path):
    config = _dataset_config(tmp_path)
    dataset = MCLSRDataset.create_from_config(config)
    train, _, _ = dataset.get_samplers()
    batch = BasicBatchProcessor()([train[0]])
    torch.manual_seed(7)
    model = BaseModel.create_from_config(
        {
            'type': 'mclsr',
            'sequence_prefix': 'item',
            'user_prefix': 'user',
            'labels_prefix': 'labels',
            'eval_top_k': 10,
            'embedding_dim': 8,
            'num_graph_layers': 0,
            'use_graph': False,
            'dropout': 0.0,
        },
        **dataset.meta,
    )
    outputs = model(batch)
    outputs['combined_representation'].square().sum().backward()
    gradient = model._position_embeddings.weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient[19].norm().item() > 0


@pytest.mark.parametrize('field', ['max_history_length', 'files_sha256'])
def test_v2_manifest_cannot_silently_fall_back_to_legacy(tmp_path, field):
    config = _dataset_config(tmp_path)
    manifest_path = tmp_path / 'Tiny' / 'preprocessing.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest[field]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        MCLSRDataset.create_from_config(config)


def test_v2_manifest_rejects_changed_file_contents(tmp_path):
    config = _dataset_config(tmp_path)
    train_file = tmp_path / 'Tiny' / 'train_mclsr.txt'
    train_file.write_text(train_file.read_text().replace(' 25\n', ' 24\n'))
    with pytest.raises(ValueError, match='hash'):
        MCLSRDataset.create_from_config(config)


def test_partial_preparation_cannot_be_loaded_as_legacy(tmp_path):
    config = _dataset_config(tmp_path)
    (tmp_path / 'Tiny' / 'preprocessing.json').unlink()
    (tmp_path / 'Tiny' / '.preprocessing-in-progress').touch()
    with pytest.raises(ValueError, match='incomplete'):
        MCLSRDataset.create_from_config(config)
