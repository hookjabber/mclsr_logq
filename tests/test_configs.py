"""Validate every training config without touching data or GPUs.

Catches a broken config in seconds instead of hours into a run:
JSON must parse, and every component `type` must exist in the registry.
Run directly (`python tests/test_configs.py`) or via pytest.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from irec.callbacks import BaseCallback  # noqa: E402
from irec.dataloader import BaseDataloader  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.loss import BaseLoss  # noqa: E402
from irec.metric import BaseMetric  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.optimizer import BaseOptimizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
MAINTAINED = sorted(
    glob.glob(os.path.join(ROOT, 'configs/train/grid/*.json'))
    + glob.glob(os.path.join(ROOT, 'configs/train/*.json'))
)
LEGACY = sorted(glob.glob(os.path.join(ROOT, 'configs/train/legacy/*.json')))

TOP_KEYS = ('experiment_name', 'dataset', 'dataloader', 'model', 'optimizer', 'loss', 'callback')


def _loss_configs(loss_cfg):
    return loss_cfg.get('losses', [loss_cfg])


def test_legacy_configs_parse():
    assert LEGACY, 'legacy configs missing'
    for path in LEGACY:
        with open(path) as f:
            json.load(f)


def test_maintained_configs_are_valid():
    assert MAINTAINED, 'maintained configs missing'
    seen_names = {}
    for path in MAINTAINED:
        rel = os.path.relpath(path, ROOT)
        with open(path) as f:
            cfg = json.load(f)

        for key in TOP_KEYS:
            assert key in cfg, f'{rel}: missing top-level key `{key}`'

        name = cfg['experiment_name']
        assert name not in seen_names, (
            f'{rel}: duplicate experiment_name `{name}` (also in {seen_names.get(name)})'
        )
        seen_names[name] = rel

        assert cfg['dataset']['type'] in BaseDataset._subclasses, (
            f'{rel}: unknown dataset type `{cfg["dataset"]["type"]}`'
        )
        assert cfg['model']['type'] in BaseModel._subclasses, (
            f'{rel}: unknown model type `{cfg["model"]["type"]}`'
        )
        assert cfg['optimizer']['type'] in BaseOptimizer._subclasses, rel

        for dl_cfg in cfg['dataloader'].values():
            assert dl_cfg['type'] in BaseDataloader._subclasses, rel

        for loss_cfg in _loss_configs(cfg['loss']):
            assert loss_cfg['type'] in BaseLoss._subclasses, (
                f'{rel}: unknown loss type `{loss_cfg["type"]}`'
            )

        for cb_cfg in cfg['callback'].get('callbacks', [cfg['callback']]):
            assert cb_cfg['type'] in BaseCallback._subclasses, (
                f'{rel}: unknown callback type `{cb_cfg["type"]}`'
            )
            for metric_cfg in cb_cfg.get('metrics', {}).values():
                assert metric_cfg['type'] in BaseMetric._subclasses, (
                    f'{rel}: unknown metric type `{metric_cfg["type"]}`'
                )


if __name__ == '__main__':
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith('test_'):
            fn()
            print(f'{fn_name}: OK')
    print(f'\nALL CONFIG TESTS PASSED ({len(MAINTAINED)} maintained + {len(LEGACY)} legacy)')
