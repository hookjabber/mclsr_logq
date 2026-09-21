from irec.utils import MetaParent

import copy
import math
import torch
import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):
    def __call__(self, inputs):
        return inputs


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
        self._log_vars = nn.ParameterList([
            nn.Parameter(torch.tensor(-math.log(w), dtype=torch.float32))
            for w, flag in zip(weights, learn) if flag
        ])
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


class FpsLoss(TorchLoss, config_name='fps'):
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau,
        normalize_embeddings=False,
        use_mean=True,
        scheme='symmetric',
        similarity='dot',
        output_prefix=None,
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(
            reduction='mean' if use_mean else 'sum',
        )
        self._normalize_embeddings = normalize_embeddings
        self._scheme = scheme
        self._similarity = similarity
        self._output_prefix = output_prefix
        if self._scheme not in ('symmetric', 'cross_only', 'paper'):
            raise ValueError(f'Unknown fps scheme `{self._scheme}`')
        if self._similarity not in ('dot', 'euclidean'):
            raise ValueError(f'Unknown fps similarity `{self._similarity}`')
        # cosine = dot over normalized embeddings; normalization, when
        # enabled, rescales the inputs in euclidean mode too (no maintained
        # config combines normalize with euclidean)

    def _pairwise_scores(self, queries, candidates):
        if self._similarity == 'euclidean':
            # exact mode: the default mm-based path is numerically asymmetric
            # on large batches (GPU), breaking the mirrored-diagonal invariant
            distances = torch.cdist(
                queries, candidates,
                compute_mode='donot_use_mm_for_euclid_dist',
            )
            return -distances ** 2 / self._tau
        return torch.mm(queries, candidates.T) / self._tau

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_embeddings_prefix=config['fst_embeddings_prefix'],
            snd_embeddings_prefix=config['snd_embeddings_prefix'],
            tau=config.get('temperature', 1.0),
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            scheme=config.get('scheme', 'symmetric'),
            similarity=config.get('similarity', 'dot'),
            output_prefix=config.get('output_prefix')
        )

    def forward(self, inputs):
        fst_embeddings = inputs[
            self._fst_embeddings_prefix
        ]  # (x, embedding_dim)
        snd_embeddings = inputs[
            self._snd_embeddings_prefix
        ]  # (x, embedding_dim)

        if self._scheme in ('cross_only', 'paper'):
            if self._normalize_embeddings:
                fst_embeddings = torch.nn.functional.normalize(
                    fst_embeddings, p=2, dim=-1, eps=1e-6,
                )
                snd_embeddings = torch.nn.functional.normalize(
                    snd_embeddings, p=2, dim=-1, eps=1e-6,
                )
            if self._scheme == 'cross_only':
                # B x B: anchors = fst view, candidates = snd view only
                # (one negative per other sample)
                scores = self._pairwise_scores(fst_embeddings, snd_embeddings)
            else:
                # paper eq. 8: anchors = fst view only; candidates = ALL
                # snd-view rows + other fst-view rows -> B x (2B-1) after
                # masking the anchor's own fst column
                candidates = torch.cat((snd_embeddings, fst_embeddings), dim=0)
                scores = self._pairwise_scores(fst_embeddings, candidates)
                n = fst_embeddings.shape[0]
                idx = torch.arange(n, device=scores.device)
                scores[idx, n + idx] = -1e12  # own fst copy is not a negative
            labels = torch.arange(scores.shape[0], device=scores.device)
            loss = self._loss_function(scores, labels)
            if self._output_prefix is not None:
                inputs[self._output_prefix] = loss.cpu().item()
            return loss

        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings),
            dim=0,
        )  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings,
                p=2,
                dim=-1,
                eps=1e-6,
            )  # (2 * x, embedding_dim)

        similarity_scores = self._pairwise_scores(
            combined_embeddings, combined_embeddings,
        )  # (2 * x, 2 * x)

        positive_samples = torch.cat(
            (
                torch.diag(similarity_scores, batch_size),
                torch.diag(similarity_scores, -batch_size),
            ),
            dim=0,
        ).reshape(2 * batch_size, 1)  # (2 * x, 1)
        assert torch.allclose(
            torch.diag(similarity_scores, batch_size),
            torch.diag(similarity_scores, -batch_size),
        )

        mask = torch.ones(
            2 * batch_size,
            2 * batch_size,
            dtype=torch.bool,
            device=similarity_scores.device,
        )  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(
            2 * batch_size,
            -1,
        )  # (2 * x, 2 * x - 2)

        labels = (
            torch.zeros(2 * batch_size).to(positive_samples.device).long()
        )  # (2 * x)
        logits = torch.cat(
            (positive_samples, negative_samples),
            dim=1,
        )  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SASRecLoss(TorchLoss, config_name='sasrec'):

    def __init__(
            self,
            positive_prefix,
            negative_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_scores = inputs[self._positive_prefix]  # (x)
        negative_scores = inputs[self._negative_prefix]  # (x)
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SamplesSoftmaxLoss(TorchLoss, config_name='sampled_softmax'):
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        queries_embeddings = inputs[
            self._queries_prefix
        ]  # (batch_size, embedding_dim)
        positive_embeddings = inputs[
            self._positive_prefix
        ]  # (batch_size, embedding_dim)
        negative_embeddings = inputs[
            self._negative_prefix
        ]  # (num_negatives, embedding_dim) or (batch_size, num_negatives, embedding_dim)

        # b -- batch_size, d -- embedding_dim
        positive_scores = torch.einsum(
            'bd,bd->b',
            queries_embeddings,
            positive_embeddings,
        ).unsqueeze(-1)  # (batch_size, 1)

        if negative_embeddings.dim() == 2:  # (num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,nd->bn',
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        else:
            assert (
                negative_embeddings.dim() == 3
            )  # (batch_size, num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,bnd->bn',
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        all_scores = torch.cat(
            [positive_scores, negative_scores],
            dim=1,
        )  # (batch_size, 1 + num_negatives)

        logits = torch.log_softmax(
            all_scores,
            dim=1,
        )  # (batch_size, 1 + num_negatives)
        loss = (-logits)[:, 0]  # (batch_size)
        loss = loss.mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
