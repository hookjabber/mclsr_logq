import math

import pytest
import torch

from irec.loss.base import UncertaintyCompositeLoss
from irec.optimizer.base import BasicOptimizer


@pytest.mark.parametrize('include_extra', [False, True])
@pytest.mark.parametrize('threshold', [None, 0.0, 0.1, 10.0])
def test_clipping_uses_all_optimized_parameters_with_one_global_norm(include_extra, threshold):
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(1.0)
    extra = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor(7.0, dtype=torch.float64))
    optimizer = BasicOptimizer.create_from_config(
        {'optimizer': {'type': 'sgd', 'lr': 0.2}, 'clip_grad_threshold': threshold},
        model=model,
        extra_parameters=[extra, unused] if include_extra else [],
    )
    loss = 3 * model.weight.sum() + (4 * extra if include_extra else 0)
    optimizer.step(loss)
    norm = 5.0 if include_extra else 3.0
    scale = 1.0 if threshold is None else min(1.0, threshold / (norm + 1e-6))
    assert model.weight.grad.item() == pytest.approx(3 * scale)
    assert model.weight.item() == pytest.approx(1 - 0.2 * 3 * scale)
    if include_extra:
        assert extra.grad.item() == pytest.approx(4 * scale)
        assert extra.item() == pytest.approx(2 - 0.2 * 4 * scale)
    assert unused.grad is None
    assert unused.item() == 7.0


def test_clipping_respects_optimizer_groups_instead_of_unoptimized_model_parameters():
    model = torch.nn.Linear(1, 1, bias=True).double()
    extra = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    torch_optimizer = torch.optim.SGD([{'params': [model.weight], 'lr': 0.2}, {'params': [extra], 'lr': 0.4}])
    optimizer = BasicOptimizer(model, torch_optimizer, clip_grad_threshold=0.1)
    loss = 3 * model.weight.sum() + 4 * extra + 12 * model.bias.sum()
    optimizer.step(loss)
    scale = 0.1 / (5.0 + 1e-6)
    assert model.weight.grad.item() == pytest.approx(3 * scale)
    assert extra.grad.item() == pytest.approx(4 * scale)
    assert model.bias.grad.item() == 12.0


class _SquaredPrediction(torch.nn.Module):
    def forward(self, inputs):
        return inputs['prediction'].square().sum()


def test_uncertainty_loss_weights_are_clipped_together_with_model_gradients():
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(2.0)
    objective = UncertaintyCompositeLoss([_SquaredPrediction()], [0.05], [True]).double()
    optimizer = BasicOptimizer.create_from_config(
        {'optimizer': {'type': 'sgd', 'lr': 0.01}, 'clip_grad_threshold': 0.1},
        model=model,
        extra_parameters=objective.parameters(),
    )
    loss = objective({'prediction': model.weight})
    parameters = [model.weight, *objective.parameters()]
    raw_gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    norm = math.sqrt(sum(gradient.square().sum().item() for gradient in raw_gradients))
    scale = min(1.0, 0.1 / (norm + 1e-6))
    optimizer.step(loss)
    for parameter, gradient in zip(parameters, raw_gradients):
        torch.testing.assert_close(parameter.grad, gradient * scale)
