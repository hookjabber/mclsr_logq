import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from irec.dataset import BaseDataset
import irec.dataset.graph as graph_module


ROOT = Path(__file__).resolve().parents[1]


def _write_v2_inputs(directory, sequences):
    directory.mkdir(exist_ok=True)
    windows = [
        (user_id, items[max(0, target_index - 20) : target_index + 1])
        for user_id, items in sequences
        for target_index in range(1, len(items))
    ]
    rows_by_file = {
        'train_sasrec.txt': sequences,
        'train_mclsr.txt': windows,
        'valid_history.txt': [(5, [1, 2, 3, 4])],
        'valid_target.txt': [(5, [1])],
        'test_history.txt': [(6, [1, 2, 3, 4])],
        'test_target.txt': [(6, [2])],
    }
    hashes = {}
    for filename, rows in rows_by_file.items():
        payload = ''.join(' '.join(map(str, [user_id, *items])) + '\n' for user_id, items in rows).encode()
        (directory / filename).write_bytes(payload)
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    (directory / 'preprocessing.json').write_text(
        json.dumps(
            {
                'protocol': 'windowed-v2',
                'max_history_length': 20,
                'num_users': 6,
                'num_items': 4,
                'files_sha256': hashes,
            }
        )
    )


def _profile(directory, mode):
    config = json.loads((ROOT / f'configs/train/validation/clothing_v2_{mode}_l00.json').read_text())
    config['dataset']['dataset']['path_to_data_dir'] = str(directory.parent)
    config['dataset']['dataset']['name'] = directory.name
    config['dataset']['graph_dir_path'] = str(directory)
    return config


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))
    specification = importlib.util.spec_from_file_location(
        'release_graph_diagnostic', ROOT / 'scripts/diagnose_feature_graphs.py'
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    def graph_only_view(config, checkpoint):
        BaseDataset.create_from_config(config['dataset'])
        return {}

    monkeypatch.setattr(module, 'view_similarity', graph_only_view)
    monkeypatch.setattr(module, 'loss_curves', lambda experiment_name: {})
    return module


def _run_diagnostic(module, config, tmp_path, monkeypatch, capsys):
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['diagnose_feature_graphs.py', '--params', str(params), '--checkpoint', 'unused'])
    module.main()
    return json.loads(capsys.readouterr().out)['graph_overlap']


@pytest.mark.parametrize('mode', ['binary', 'event_count'])
@pytest.mark.parametrize('neighbors', [None, 1, 50])
def test_graph_diagnostic_uses_profile_cache(tmp_path, monkeypatch, capsys, diagnostic, mode, neighbors):
    directory = tmp_path / 'data'
    _write_v2_inputs(directory, [(1, [1, 1, 2]), (2, [2, 3, 3]), (3, [3, 4, 4])])
    config = _profile(directory, mode)
    config['dataset']['neighborhood_size'] = neighbors
    BaseDataset.create_from_config(config['dataset'])

    report = _run_diagnostic(diagnostic, config, tmp_path, monkeypatch, capsys)

    assert report['user-user']['rows'] == 8
    assert report['item-item']['rows'] == 6


def test_graph_diagnostic_revalidates_stale_cache_before_reporting(tmp_path, monkeypatch, capsys, diagnostic):
    directory = tmp_path / 'data'
    _write_v2_inputs(directory, [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])])
    config = _profile(directory, 'binary')
    config['dataset']['neighborhood_size'] = 1
    BaseDataset.create_from_config(config['dataset'])
    stale_report = diagnostic.graph_overlap(directory, k=1)
    _write_v2_inputs(directory, [(1, [1, 2, 3]), (2, [1, 2]), (3, [3, 4]), (4, [3, 4])])

    report = _run_diagnostic(diagnostic, config, tmp_path, monkeypatch, capsys)

    for entity in ('user-user', 'item-item'):
        key = 'mean_jaccard_with_top1_of_2hop'
        assert stale_report[entity][key] < 1.0
        assert diagnostic.graph_overlap(directory, k=1)[entity][key] == 1.0
        assert report[entity][key] == 1.0


@pytest.mark.parametrize('user_ids,expected', [([1, 1, 2], 0.0), ([1], None)])
def test_view_similarity_compares_distinct_entities(tmp_path, monkeypatch, user_ids, expected):
    specification = importlib.util.spec_from_file_location(
        'graph_view_diagnostic', ROOT / 'scripts/diagnose_feature_graphs.py'
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    identities = torch.tensor(user_ids)
    vectors = torch.eye(2)[identities - 1]

    class ViewModel(torch.nn.Module):
        def forward(self, inputs):
            return {
                'user_ids': identities,
                'item_ids': identities,
                **{
                    key: vectors
                    for key in (
                        'user_graph_user_embeddings',
                        'common_graph_user_embeddings',
                        'item_graph_item_embeddings',
                        'common_graph_item_embeddings',
                        'sequential_representation',
                        'graph_representation',
                    )
                },
            }

    monkeypatch.setattr(module, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(
        module.BaseDataset,
        'create_from_config',
        lambda config: SimpleNamespace(meta={}, get_samplers=lambda: (None, None, None)),
    )
    monkeypatch.setattr(module.BaseDataloader, 'create_from_config', lambda *arguments, **kwargs: [{}])
    monkeypatch.setattr(module.BaseModel, 'create_from_config', lambda *arguments, **kwargs: ViewModel())
    monkeypatch.setattr(module.torch, 'load', lambda *arguments, **kwargs: {})

    report = module.view_similarity({'dataset': {}, 'dataloader': {'train': {}}, 'model': {}}, tmp_path / 'unused')
    for values in report.values():
        assert values['cos_same_entity'] == pytest.approx(1.0)
        assert values['cos_other_entities'] == expected
