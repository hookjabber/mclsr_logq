"""Loss base classes and the composite objectives (paper eq. 15; learned-weight variant)."""

from irec.utils import MetaParent
import copy
import math
import torch
import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class CompositeLoss(TorchLoss, config_name='composite'):
    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        # a registered container: `.to(device)` and `.parameters()` reach the sub-losses
        # (their count tables are buffers, learned loss weights are parameters)
        self._losses = nn.ModuleList(losses)
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in copy.deepcopy(config)['losses']:
            weight = loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(
            losses=losses,
            weights=weights,
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss


class UncertaintyCompositeLoss(TorchLoss, config_name='composite_uncertainty'):
    """
    Composite loss with learned weights (Kendall, Gal & Cipolla 2018): for every
    term flagged `learn_weight`, weight = exp(-s) with a learnable s = log sigma^2
    and the regulariser +s; s is initialised so that exp(-s) equals the configured
    weight. Terms without the flag keep their fixed weight (the retrieval loss
    stays fixed as the anchor). Effective weights are written to
    inputs['weight/<output_prefix>'] for logging.
    """

    def __init__(self, losses, weights, learn, output_prefix=None):
        super().__init__()
        self._losses = nn.ModuleList(losses)
        self._weights = weights
        self._learn = learn
        self._log_vars = nn.ParameterList(
            [nn.Parameter(torch.tensor(-math.log(w), dtype=torch.float32)) for w, flag in zip(weights, learn) if flag]
        )
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses, weights, learn = [], [], []
        for loss_cfg in copy.deepcopy(config)['losses']:
            weights.append(loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0)
            learn.append(bool(loss_cfg.pop('learn_weight', False)))
            losses.append(BaseLoss.create_from_config(loss_cfg))
        return cls(losses=losses, weights=weights, learn=learn, output_prefix=config.get('output_prefix'))

    def forward(self, inputs):
        total = 0.0
        k = 0
        for loss, weight, flag in zip(self._losses, self._weights, self._learn):
            value = loss(inputs)
            if flag:
                s = self._log_vars[k]
                k += 1
                effective = torch.exp(-s)
                total = total + effective * value + s
                prefix = getattr(loss, '_output_prefix', None)
                if prefix:
                    inputs['weight/' + prefix] = float(effective.detach().cpu())
            else:
                total = total + weight * value
        if self._output_prefix is not None:
            inputs[self._output_prefix] = total.cpu().item()
        return total
