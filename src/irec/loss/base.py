from irec.utils import (
    MetaParent,
    maybe_to_list,
)

import copy
import torch
import torch.nn as nn
import pickle
import os
import logging


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
        self._losses = losses
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


class FpsLoss(TorchLoss, config_name='fps'):
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau,
        normalize_embeddings=False,
        use_mean=True,
        scheme='symmetric',
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
        self._output_prefix = output_prefix
        if self._scheme not in ('symmetric', 'cross_only', 'paper'):
            raise ValueError(f'Unknown fps scheme `{self._scheme}`')

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_embeddings_prefix=config['fst_embeddings_prefix'],
            snd_embeddings_prefix=config['snd_embeddings_prefix'],
            tau=config.get('temperature', 1.0),
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            scheme=config.get('scheme', 'symmetric'),
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
                scores = torch.mm(fst_embeddings, snd_embeddings.T) / self._tau
            else:
                # paper eq. 8: anchors = fst view only; candidates = ALL
                # snd-view rows + other fst-view rows -> B x (2B-1) after
                # masking the anchor's own fst column
                candidates = torch.cat((snd_embeddings, fst_embeddings), dim=0)
                scores = torch.mm(fst_embeddings, candidates.T) / self._tau
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

        similarity_scores = (
            torch.mm(combined_embeddings, combined_embeddings.T) / self._tau
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


class FpsLogQLoss(TorchLoss, config_name='fps_logq'):
    """
    LogQ-corrected two-view in-batch contrastive loss.

    This is the LogQ variant of FpsLoss for losses such as L_IL, where the
    candidates are other samples/users in the same batch rather than explicit
    item negatives. The configured ids_prefix defines which sample id supplies
    q(id) for the correction table. By default q(id) is converted to the
    probability that this id appears at least once among B-1 in-batch negatives.

    With `leave_own_out=True` the correction uses the leave-anchor-out
    distribution q'_anchor(v) = count(v) / (S - count(anchor)): false-negative
    masking removes the anchor's own id from the candidate pool, so negatives
    are effectively drawn from q', not from the marginal q.
    """
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        ids_prefix,
        path_to_counts,
        tau,
        logq_lambda=1.0,
        logq_probability_mode='inbatch_negative',
        normalize_embeddings=False,
        use_mean=True,
        mask_false_negatives=True,
        leave_own_out=False,
        scheme='symmetric',
        output_prefix=None,
        num_draws_prefix=None,
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._ids_prefix = ids_prefix
        self._tau = tau
        self._logq_lambda = logq_lambda
        self._logq_probability_mode = logq_probability_mode
        self._num_draws_prefix = num_draws_prefix
        self._loss_function = nn.CrossEntropyLoss(
            reduction='mean' if use_mean else 'sum',
        )
        self._normalize_embeddings = normalize_embeddings
        self._mask_false_negatives = mask_false_negatives
        self._leave_own_out = leave_own_out
        self._scheme = scheme
        self._output_prefix = output_prefix

        if self._scheme not in ('symmetric', 'cross_only'):
            raise ValueError(f'Unknown fps_logq scheme `{self._scheme}`')

        if self._leave_own_out and not self._mask_false_negatives:
            raise ValueError(
                'FpsLogQLoss `leave_own_out` requires `mask_false_negatives`: '
                "q' is the negative distribution induced by masking the anchor id"
            )

        if not os.path.exists(path_to_counts):
            raise FileNotFoundError(f"Counts file not found at {path_to_counts}")

        with open(path_to_counts, 'rb') as f:
            counts = pickle.load(f)

        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10, max=1.0)
        self.register_buffer('_prob_table', probs)
        self.register_buffer('_log_q_table', torch.log(probs))

        allowed_modes = {'sample', 'inbatch_negative'}
        if self._logq_probability_mode not in allowed_modes:
            raise ValueError(
                'FpsLogQLoss `logq_probability_mode` must be one of '
                f'{sorted(allowed_modes)}, got {self._logq_probability_mode}'
            )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        path_to_counts = config.get(
            'path_to_counts',
            config.get('path_to_item_counts'),
        )
        if path_to_counts is None:
            raise ValueError(
                'FpsLogQLoss requires `path_to_counts` in the loss config'
            )

        return cls(
            fst_embeddings_prefix=config['fst_embeddings_prefix'],
            snd_embeddings_prefix=config['snd_embeddings_prefix'],
            ids_prefix=config['ids_prefix'],
            path_to_counts=path_to_counts,
            tau=config.get('temperature', 1.0),
            logq_lambda=config.get('logq_lambda', 1.0),
            logq_probability_mode=config.get(
                'logq_probability_mode',
                'inbatch_negative',
            ),
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            mask_false_negatives=config.get('mask_false_negatives', True),
            leave_own_out=config.get('leave_own_out', False),
            scheme=config.get('scheme', 'symmetric'),
            output_prefix=config.get('output_prefix'),
            num_draws_prefix=config.get('num_draws_prefix'),
        )

    def _sample_log_q(self, ids, num_negative_draws, device):
        if self._logq_probability_mode == 'sample':
            return self._log_q_table.to(device=device)[ids]  # (B,)
        sample_probs = self._prob_table.to(device=device)[ids]
        if num_negative_draws <= 0:
            sample_q = torch.full_like(sample_probs, 1e-10)
        else:
            sample_q = 1.0 - torch.pow(1.0 - sample_probs, num_negative_draws)
            sample_q = torch.clamp(sample_q, min=1e-10)
        return torch.log(sample_q)  # (B,)

    def _candidate_log_q(self, ids, num_negative_draws, device):
        sample_log_q = self._sample_log_q(ids, num_negative_draws, device)
        return torch.cat(
            (sample_log_q, sample_log_q),
            dim=0,
        )  # (2 * B,)

    def _loo_log_q_square(self, ids, num_negative_draws, device):
        # (B, B): row = anchor, column = candidate; q'_a(j) = p_j / (1 - p_a)
        sample_probs = self._prob_table.to(device=device)[ids]  # (B,)
        anchor_keep = torch.clamp(1.0 - sample_probs, min=1e-10)
        probs = sample_probs.unsqueeze(0) / anchor_keep.unsqueeze(1)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        if self._logq_probability_mode == 'sample':
            return torch.log(probs)
        if num_negative_draws <= 0:
            sample_q = torch.full_like(probs, 1e-10)
        else:
            sample_q = 1.0 - torch.pow(
                torch.clamp(1.0 - probs, min=0.0),
                num_negative_draws,
            )
            sample_q = torch.clamp(sample_q, min=1e-10)
        return torch.log(sample_q)

    def _candidate_log_q_leave_own_out(self, ids, num_negative_draws, device):
        # q'_anchor(v) = count(v) / (S - count(anchor)) = p(v) / (1 - p(anchor)):
        # the anchor's own id is masked out of the pool, every draw comes from q'.
        sample_probs = self._prob_table.to(device=device)[ids]         # (B,)
        candidate_probs = torch.cat((sample_probs, sample_probs))      # (2 * B,)
        anchor_keep = torch.clamp(1.0 - candidate_probs, min=1e-10)    # (2 * B,)
        probs = candidate_probs.unsqueeze(0) / anchor_keep.unsqueeze(1)
        probs = torch.clamp(probs, min=1e-10, max=1.0)  # (2 * B, 2 * B), row = anchor

        if self._logq_probability_mode == 'sample':
            return torch.log(probs)

        if num_negative_draws <= 0:
            sample_q = torch.full_like(probs, 1e-10)
        else:
            sample_q = 1.0 - torch.pow(
                torch.clamp(1.0 - probs, min=0.0),
                num_negative_draws,
            )
            sample_q = torch.clamp(sample_q, min=1e-10)
        return torch.log(sample_q)

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (B, D)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (B, D)
        ids = inputs[self._ids_prefix]                        # (B,)

        if fst_embeddings.shape != snd_embeddings.shape:
            raise ValueError(
                'FpsLogQLoss expects both embedding tensors to have the same shape'
            )

        batch_size = fst_embeddings.shape[0]
        if ids.shape[0] != batch_size:
            raise ValueError(
                f'FpsLogQLoss got {ids.shape[0]} ids for batch size {batch_size}'
            )

        if self._scheme == 'cross_only':
            # B x B: anchors = fst view, candidates = snd view only
            # (one negative per other sample, not two)
            if self._normalize_embeddings:
                fst_embeddings = torch.nn.functional.normalize(
                    fst_embeddings, p=2, dim=-1, eps=1e-6,
                )
                snd_embeddings = torch.nn.functional.normalize(
                    snd_embeddings, p=2, dim=-1, eps=1e-6,
                )
            all_scores = torch.mm(fst_embeddings, snd_embeddings.T) / self._tau
            device = all_scores.device
            ids = ids.to(device=device)
            if self._num_draws_prefix is not None:
                num_negative_draws = int(inputs[self._num_draws_prefix])
            else:
                num_negative_draws = batch_size - 1

            if self._leave_own_out:
                candidate_log_q = self._loo_log_q_square(
                    ids, num_negative_draws, device,
                )  # (B, B)
                all_scores = all_scores - self._logq_lambda * candidate_log_q
                positive_log_q = candidate_log_q.diagonal()
            else:
                candidate_log_q = self._sample_log_q(
                    ids, num_negative_draws, device,
                )  # (B,)
                all_scores = all_scores - self._logq_lambda * candidate_log_q.unsqueeze(0)
                positive_log_q = candidate_log_q

            # the paired positive (diagonal) is observed, not sampled
            all_scores.diagonal().add_(self._logq_lambda * positive_log_q)

            if self._mask_false_negatives:
                false_negative_mask = ids.unsqueeze(0) == ids.unsqueeze(1)
                false_negative_mask.fill_diagonal_(False)
                all_scores = all_scores.masked_fill(false_negative_mask, -1e12)

            labels = torch.arange(batch_size, device=device)
            loss = self._loss_function(all_scores, labels)
            if self._output_prefix is not None:
                inputs[self._output_prefix] = loss.cpu().item()
            return loss

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings),
            dim=0,
        )  # (2 * B, D)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings,
                p=2,
                dim=-1,
                eps=1e-6,
            )

        all_scores = (
            torch.mm(combined_embeddings, combined_embeddings.T) / self._tau
        )  # (2 * B, 2 * B)

        device = all_scores.device
        ids = ids.to(device=device)
        if self._num_draws_prefix is not None:
            num_negative_draws = int(inputs[self._num_draws_prefix])
        else:
            num_negative_draws = batch_size - 1

        if self._leave_own_out:
            candidate_log_q = self._candidate_log_q_leave_own_out(
                ids, num_negative_draws, device,
            )  # (2 * B, 2 * B), row = anchor, column = candidate
            all_scores = all_scores - self._logq_lambda * candidate_log_q
        else:
            candidate_log_q = self._candidate_log_q(
                ids, num_negative_draws, device,
            )  # (2 * B,)
            all_scores = all_scores - self._logq_lambda * candidate_log_q.unsqueeze(0)

        num_views = 2 * batch_size
        row_indices = torch.arange(num_views, device=device)
        positive_indices = (row_indices + batch_size) % num_views

        # The paired positive is the observed sample, not a sampled negative.
        if self._leave_own_out:
            positive_log_q = candidate_log_q[row_indices, positive_indices]
        else:
            positive_log_q = candidate_log_q[positive_indices]
        all_scores[row_indices, positive_indices] += (
            self._logq_lambda * positive_log_q
        )

        invalid_mask = torch.eye(
            num_views,
            dtype=torch.bool,
            device=device,
        )
        if self._mask_false_negatives:
            candidate_ids = torch.cat((ids, ids), dim=0)  # (2 * B,)
            false_negative_mask = (
                candidate_ids.unsqueeze(0) == candidate_ids.unsqueeze(1)
            )
            false_negative_mask[row_indices, positive_indices] = False
            false_negative_mask.fill_diagonal_(False)
            invalid_mask |= false_negative_mask

        all_scores = all_scores.masked_fill(invalid_mask, -1e12)

        loss = self._loss_function(all_scores, positive_indices) / 2

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


class MCLSRLoss(TorchLoss, config_name='mclsr'):
    def __init__(
        self,
        all_scores_prefix,
        mask_prefix,
        normalize_embeddings=False,
        tau=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._all_scores_prefix = all_scores_prefix
        self._mask_prefix = mask_prefix
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        self._tau = tau

    def forward(self, inputs):
        all_scores = inputs[
            self._all_scores_prefix
        ]  # (batch_size, batch_size, seq_len)
        mask = inputs[self._mask_prefix]  # (batch_size)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positive_mask = torch.eye(batch_size, device=mask.device).bool()

        positive_scores = all_scores[positive_mask]  # (batch_size, seq_len)
        negative_scores = torch.reshape(
            all_scores[~positive_mask],
            shape=(batch_size, batch_size - 1, seq_len),
        )  # (batch_size, batch_size - 1, seq_len)
        assert torch.allclose(all_scores[0, 1], negative_scores[0, 0])
        assert torch.allclose(all_scores[-1, -2], negative_scores[-1, -1])
        assert torch.allclose(all_scores[0, 0], positive_scores[0])
        assert torch.allclose(all_scores[-1, -1], positive_scores[-1])

        # Maybe try mean over sequence TODO
        loss = torch.sum(
            torch.log(
                torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores),
            ),
        )  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
    
class MCLSRLogqLoss(TorchLoss, config_name='mclsr_logq_special'):
    """
    LogQ-corrected Sampled Softmax Loss for MCLSR model.
    Implements sampling-bias correction: s_c(x, y) = s(x, y) - lambda * log(p_j)
    
    This adjustment compensates for non-uniform negative sampling (e.g., popularity-based),
    preventing the model from over-penalizing popular items.
    """
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        positive_ids_prefix,
        negative_ids_prefix,
        path_to_item_counts,
        logq_lambda=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._negative_ids_prefix = negative_ids_prefix
        self._output_prefix = output_prefix
        self._logq_lambda = logq_lambda

        # Load global item frequencies to calculate sampling probabilities (p_j)
        if not os.path.exists(path_to_item_counts):
            raise FileNotFoundError(f"Item counts file not found at {path_to_item_counts}")

        with open(path_to_item_counts, 'rb') as f:
            counts = pickle.load(f)
        
        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        
        # Calculate log-probabilities. 
        # Clamp used for numerical stability to avoid log(0) resulting in NaN.
        probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10)
        log_q = torch.log(probs)
        
        # register_buffer ensures the lookup table is moved to the correct 
        # device (GPU/CPU) automatically during training.
        self.register_buffer('_log_q_table', log_q)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        """Factory method to initialize loss from JSON configuration."""
        return cls(
            queries_prefix=config['queries_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            positive_ids_prefix=config['positive_ids_prefix'],
            negative_ids_prefix=config['negative_ids_prefix'],
            path_to_item_counts=config['path_to_item_counts'],
            logq_lambda=config.get('logq_lambda', 1.0),
            output_prefix=config.get('output_prefix')
        )

    def forward(self, inputs):
        # 1. Extract embeddings and item IDs
        queries = inputs[self._queries_prefix]           # (Batch, Dim)
        pos_embs = inputs[self._positive_prefix]         # (Batch, Dim)
        neg_embs = inputs[self._negative_prefix]         # (Batch, NumNegs, Dim)
        
        pos_ids = inputs[self._positive_ids_prefix]      # (Batch)
        neg_ids = inputs[self._negative_ids_prefix]      # (Batch, NumNegs)

        # Device synchronization check
        if self._log_q_table.device != queries.device:
            self._log_q_table = self._log_q_table.to(queries.device)

        # 2. Compute raw scores (Dot Product)
        # Using einsum for efficient multiplication of 2D queries and 3D negatives
        pos_scores = torch.einsum('bd,bd->b', queries, pos_embs).unsqueeze(-1) # (B, 1)
        neg_scores = torch.einsum('bd,bnd->bn', queries, neg_embs)             # (B, N)

        # 3. False Negative Masking
        # Neutralize cases where the sampled negative item is actually the target item
        false_negative_mask = (pos_ids.unsqueeze(1) == neg_ids)
        neg_scores = neg_scores.masked_fill(false_negative_mask, -1e12)

        # 4. Apply LogQ Correction
        # Correction term: score = score - lambda * log(p_j)
        log_q_pos = self._log_q_table[pos_ids].unsqueeze(-1) # (B, 1)
        log_q_neg = self._log_q_table[neg_ids]               # (B, N)
        
        pos_scores = pos_scores - (self._logq_lambda * log_q_pos)
        neg_scores = neg_scores - (self._logq_lambda * log_q_neg)

        # 5. Final Softmax Reranking
        # Concatenate scores and compute cross-entropy over the sampled items
        all_scores = torch.cat([pos_scores, neg_scores], dim=1) # (B, 1+N)
        loss = -torch.log_softmax(all_scores, dim=1)[:, 0]
        
        final_loss = loss.mean()
        if self._output_prefix:
            inputs[self._output_prefix] = final_loss.cpu().item()
            
        return final_loss


class MCLSRLogqInBatchLoss(TorchLoss, config_name='mclsr_logq_inbatch'):
    """
    LogQ-corrected In-Batch Sampled Softmax Loss for MCLSR model.

    Uses in-batch negatives: positive items of other users in the batch serve as negatives.
    This naturally produces a popularity-proportional sampling distribution,
    which LogQ correction precisely compensates.

    LogQ correction is applied only to negatives (not to the positive).

    With `leave_own_out=True` the correction uses the leave-anchor-out
    distribution q'_i(j) = q(j) / (1 - q(pos_i)): the false-negative mask
    removes the anchor's own positive item from the candidate pool, so
    in-batch negatives are effectively drawn from q', not from the marginal q.
    """
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        positive_ids_prefix,
        path_to_item_counts,
        logq_lambda=1.0,
        leave_own_out=False,
        normalize_embeddings=False,
        temperature=1.0,
        output_prefix=None,
        user_ids_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._user_ids_prefix = user_ids_prefix
        self._output_prefix = output_prefix
        self._logq_lambda = logq_lambda
        self._leave_own_out = leave_own_out
        self._normalize_embeddings = normalize_embeddings
        self._temperature = temperature

        if not os.path.exists(path_to_item_counts):
            raise FileNotFoundError(f"Item counts file not found at {path_to_item_counts}")

        with open(path_to_item_counts, 'rb') as f:
            counts = pickle.load(f)

        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10)
        log_q = torch.log(probs)
        self.register_buffer('_log_q_table', log_q)
        self.register_buffer('_prob_table', probs)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            queries_prefix=config['queries_prefix'],
            positive_prefix=config['positive_prefix'],
            positive_ids_prefix=config['positive_ids_prefix'],
            path_to_item_counts=config['path_to_item_counts'],
            logq_lambda=config.get('logq_lambda', 1.0),
            leave_own_out=config.get('leave_own_out', False),
            normalize_embeddings=config.get('normalize_embeddings', False),
            temperature=config.get('temperature', 1.0),
            output_prefix=config.get('output_prefix'),
            user_ids_prefix=config.get('user_ids_prefix'),
        )

    def forward(self, inputs):
        queries = inputs[self._queries_prefix]       # (B, D)
        pos_embs = inputs[self._positive_prefix]     # (B, D)
        pos_ids = inputs[self._positive_ids_prefix]  # (B,)

        if self._log_q_table.device != queries.device:
            self._log_q_table = self._log_q_table.to(queries.device)
        if self._prob_table.device != queries.device:
            self._prob_table = self._prob_table.to(queries.device)

        batch_size = queries.size(0)

        if self._normalize_embeddings:
            queries = torch.nn.functional.normalize(
                queries, p=2, dim=-1, eps=1e-6,
            )
            pos_embs = torch.nn.functional.normalize(
                pos_embs, p=2, dim=-1, eps=1e-6,
            )

        # All-pairs scores: each query against all positive items in batch
        # all_scores[i, j] = query_i · pos_emb_j
        # Diagonal (i, i) = positive score, off-diagonal = in-batch negatives
        all_scores = torch.mm(queries, pos_embs.T) / self._temperature  # (B, B)

        # LogQ correction on negatives only:
        # 1) Apply correction to ALL candidates (columns)
        # 2) Undo correction on the diagonal (positives)
        log_q = self._log_q_table[pos_ids]  # (B,)
        all_scores = all_scores - self._logq_lambda * log_q.unsqueeze(0)  # (B, B)
        if self._leave_own_out:
            # log q'_i(j) = log q(j) - log(1 - q(pos_i)): row-wise constant
            # accounting for the anchor's own item being masked from the pool
            row_keep_log = torch.log(
                torch.clamp(1.0 - self._prob_table[pos_ids], min=1e-10),
            )  # (B,) = log(1 - q(pos_i))
            all_scores = all_scores + self._logq_lambda * row_keep_log.unsqueeze(1)
            all_scores.diagonal().add_(self._logq_lambda * (log_q - row_keep_log))
        else:
            all_scores.diagonal().add_(self._logq_lambda * log_q)

        # False negative masking: if pos_ids[i] == pos_ids[j] and i != j,
        # then user j's positive is actually the same item as user i's positive
        false_neg_mask = (pos_ids.unsqueeze(0) == pos_ids.unsqueeze(1))  # (B, B)
        false_neg_mask.fill_diagonal_(False)
        all_scores = all_scores.masked_fill(false_neg_mask, -1e12)

        # Same-user masking: with the leave-one-out ladder a user appears in the
        # batch multiple times, so another prefix's target is an item that same
        # user actually consumed -> a false negative the item-id mask above misses.
        if self._user_ids_prefix is not None:
            user_ids = inputs[self._user_ids_prefix].reshape(-1)  # (B,)
            same_user_mask = (user_ids.unsqueeze(0) == user_ids.unsqueeze(1))  # (B, B)
            same_user_mask.fill_diagonal_(False)
            all_scores = all_scores.masked_fill(same_user_mask, -1e12)

        # Cross-entropy: positive is the diagonal (target index i for row i)
        labels = torch.arange(batch_size, device=queries.device)
        loss = torch.nn.functional.cross_entropy(all_scores, labels)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class ContrastiveFullSoftmaxLoss(TorchLoss, config_name='contrastive_full_softmax'):
    """
    Exact full-catalog contrastive anchor.

    Batch anchors from one view are contrasted against the FULL projected
    table of the other view (all users or all items): the positive is the
    anchor's own row in the table, every other entity is a negative. No
    sampling and hence no sampling bias — this is the gold standard that the
    in-batch contrastive losses approximate. Padding (index 0) and mask
    (last index) columns are excluded from the softmax.
    """
    def __init__(
        self,
        anchors_prefix,
        table_prefix,
        ids_prefix,
        tau=1.0,
        normalize_embeddings=False,
        use_mean=True,
        output_prefix=None,
    ):
        super().__init__()
        self._anchors_prefix = anchors_prefix
        self._table_prefix = table_prefix
        self._ids_prefix = ids_prefix
        self._tau = tau
        self._normalize_embeddings = normalize_embeddings
        self._loss_function = nn.CrossEntropyLoss(
            reduction='mean' if use_mean else 'sum',
        )
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            anchors_prefix=config['anchors_prefix'],
            table_prefix=config['table_prefix'],
            ids_prefix=config['ids_prefix'],
            tau=config.get('temperature', 1.0),
            normalize_embeddings=config.get('normalize_embeddings', False),
            use_mean=config.get('use_mean', True),
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        anchors = inputs[self._anchors_prefix]        # (B, D)
        table = inputs[self._table_prefix]            # (N, D)
        ids = inputs[self._ids_prefix].reshape(-1)    # (B,)

        if self._normalize_embeddings:
            anchors = torch.nn.functional.normalize(
                anchors, p=2, dim=-1, eps=1e-6,
            )
            table = torch.nn.functional.normalize(
                table, p=2, dim=-1, eps=1e-6,
            )

        all_scores = torch.mm(anchors, table.T) / self._tau  # (B, N)
        all_scores[:, 0] = -1e12   # padding column
        all_scores[:, -1] = -1e12  # mask-token column

        loss = self._loss_function(all_scores, ids)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
