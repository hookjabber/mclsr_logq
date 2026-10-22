import importlib.util
from pathlib import Path

import pytest
import torch


def test_checkpoint_evaluation_rejects_empty_split_instead_of_reporting_nan():
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'eval_checkpoint.py'
    spec = importlib.util.spec_from_file_location('release_eval_checkpoint', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match='evaluation.*(empty|no.*(user|metric|value))'):
        module.evaluate(
            model=torch.nn.Identity(),
            dataloader=[],
            metric_configs={'recall@1': {'type': 'mclsr-recall', 'k': 1}},
            pred_prefix='predictions',
            labels_prefix='labels',
            meta={},
        )
