import importlib.util
import json
import pickle
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from irec.dataset.sequences import MCLSRDataset
from irec.loss import BaseLoss
import irec.training


ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f'release_data_{name}', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('counts_kind', ['context', 'sampler', 'disabled', 'exploratory_cli', 'confirmatory_cli'])
def test_popular_logq_validates_the_sampler_base_proposal(tmp_path, monkeypatch, counts_kind):
    config = json.loads((ROOT / 'configs/train/beauty/02_popular127_logq.json').read_text())
    _script('prepare_dataset').write_files([(1, [1, 2]), (2, [3, 3, 4])], {1, 2}, set(), set(), tmp_path)
    dataset_config = config['dataset']
    dataset_config.update(path_to_data_dir=str(tmp_path.parent), name=tmp_path.name)
    dataset_config['samplers']['num_negatives_train'] = 1
    train = MCLSRDataset.create_from_config(dataset_config).get_samplers()[0]
    target_path = tmp_path / 'item_target_counts.pkl'
    context_path = tmp_path / 'item_context_counts.pkl'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'generate_mclsr_role_counts.py',
            '--input',
            str(tmp_path / 'train_mclsr.txt'),
            '--num_items',
            '4',
            '--target_output',
            str(target_path),
            '--context_output',
            str(context_path),
        ],
    )
    _script('generate_mclsr_role_counts').main()
    loss_config = config['loss']['losses'][0]
    configured_counts = tmp_path / Path(loss_config['path_to_item_counts']).name
    with configured_counts.open('rb') as source:
        counts = pickle.load(source)['counts']
    sampler = train._negative_sampler
    assert 4 in sampler._support_items - sampler._sample_seen_items(train.dataset[0])
    assert not np.allclose(counts[sampler._item_ids] / counts.sum(), sampler._probs)
    if counts_kind == 'sampler':
        with configured_counts.open('wb') as destination:
            pickle.dump([0, 1, 1, 4, 1, 0], destination)
    if counts_kind == 'disabled':
        loss_config['logq_lambda'] = 0.0
    loss_config['path_to_item_counts'] = str(configured_counts)
    if counts_kind.endswith('_cli'):
        config['model']['eval_top_k'] = 4
        parameters = tmp_path / 'config.json'
        parameters.write_text(json.dumps(config))
        if counts_kind == 'exploratory_cli':
            command = [sys.executable, '-m', 'irec.train', '--params', str(parameters)]
        else:
            command = [
                sys.executable,
                str(ROOT / 'scripts/train_confirmatory.py'),
                '--params',
                str(parameters),
                '--seed',
                '1',
                '--max-steps',
                '1',
                '--select',
                'validation/ndcg@20',
                '--output',
                str(tmp_path / 'result.json'),
            ]
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES=''),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert 'count table does not match the sampler base proposal' in result.stderr
        assert not list(tmp_path.glob('checkpoints/*.pth'))
        assert not (tmp_path / 'result.json').exists()
        return
    objective = BaseLoss.create_from_config(config['loss'])
    if counts_kind == 'context':
        with pytest.raises(ValueError, match='count table.*sampler base proposal'):
            irec.training.validate_sampling_proposals(train, objective)
    else:
        irec.training.validate_sampling_proposals(train, objective)
        assert all(torch.isfinite(buffer).all() for buffer in objective.buffers())
