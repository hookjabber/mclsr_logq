import pytest
import torch

import irec.training
from irec.optimizer.base import BasicOptimizer


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        return {'prediction': self.weight * inputs['features']}


def _run(monkeypatch, dataloader, steps, **options):
    monkeypatch.setattr(irec.training, 'DEVICE', torch.device('cpu'))
    model = _Model()
    optimizer = BasicOptimizer(model, torch.optim.SGD(model.parameters(), lr=0.01))

    def callback(batch, step):
        steps.append(step)
        batch['validation/score'] = 0.5

    return irec.training.train(
        dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=lambda batch: batch['prediction'].square().sum(),
        callback=callback,
        epochs_threshold=1,
        **options,
    )


@pytest.mark.parametrize('best_metric', [None, []])
@pytest.mark.parametrize('budget', [{'epoch_cnt': 5}, {'step_cnt': 5}])
def test_final_state_only_training_uses_the_requested_budget(monkeypatch, best_metric, budget):
    steps = []
    best = _run(monkeypatch, [{'features': torch.ones(1)}], steps, best_metric=best_metric, **budget)
    assert steps == list(range(5))
    assert best == {}


@pytest.mark.parametrize('dataloader', [[], torch.utils.data.DataLoader([1], batch_size=2, drop_last=True)])
def test_empty_training_loader_fails_without_claiming_completion(monkeypatch, dataloader):
    steps = []
    with pytest.raises(ValueError, match='Training dataloader produced no batches'):
        _run(monkeypatch, dataloader, steps, epoch_cnt=1, best_metric='validation/score')
    assert steps == []


def test_metric_based_early_stopping_is_preserved(monkeypatch):
    steps = []
    best = _run(monkeypatch, [{'features': torch.ones(1)}], steps, step_cnt=5, best_metric='validation/score')
    assert steps == [0, 1]
    assert best['validation/score']['step'] == 0


@pytest.mark.parametrize('budget', [{'epoch_cnt': 0}, {'step_cnt': 0}])
def test_zero_training_budget_does_not_require_a_batch(monkeypatch, budget):
    steps = []
    assert _run(monkeypatch, [], steps, **budget) == {}
    assert steps == []
