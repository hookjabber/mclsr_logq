"""Retrieval losses: the prediction objective L_P of the paper (eq. 14) and its variants.

* ``InBatchLogQLoss`` (``mclsr_logq_inbatch``): in-batch sampled softmax with the three forms of the
  logQ correction (``negatives_only``, ``standard``, ``corrected``), leave-own-out q′, mixed negatives;
* ``FullSoftmaxLoss`` (``full_softmax``): the exact full-catalogue softmax, the no-sampling anchor;
* ``SampledLogQLoss`` (``mclsr_logq_special``) and ``SampledSoftmaxLoss`` (``sampled_softmax``): explicit sampled
  negatives with and without the correction;
* ``SASRecLoss`` (``sasrec``): the classic binary cross-entropy with one negative.
Formulas and the comparison with the released RecSys 2025 implementation: VERIFICATION.md.
"""

import torch

from .base import TorchLoss
from .counts import load_counts_artifact


class InBatchLogQLoss(TorchLoss, config_name='mclsr_logq_inbatch'):
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
        expected_counts_role=None,
        expected_counts_max_len=None,
        correct_positive=False,
        variant='negatives_only',
        mixed_uniform_negatives=0,
        table_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._user_ids_prefix = user_ids_prefix
        self._output_prefix = output_prefix
        self._logq_lambda = logq_lambda
        self._leave_own_out = leave_own_out
        # mixed negative sampling (Yang et al. 2020): K uniform catalog items are
        # appended to the in-batch pool (shared by the whole batch, as the in-batch
        # positives are); the proposal becomes the mixture
        #   Q_mix(v) = B/(B+K) * q(v) + K/(B+K) * 1/|V|
        # and the correction uses log Q_mix. Needs the raw item table.
        self._mixed_uniform_negatives = int(mixed_uniform_negatives)
        self._table_prefix = table_prefix
        if self._mixed_uniform_negatives > 0 and table_prefix is None:
            raise ValueError('mixed_uniform_negatives requires table_prefix (the item table)')
        if self._mixed_uniform_negatives > 0 and leave_own_out:
            raise ValueError('mixed_uniform_negatives is not defined with leave_own_out')
        # three forms of the correction:
        #   negatives_only — the positive stays in the denominator uncorrected
        #                    (Yang et al. 2020 / MNS convention; the study default);
        #   standard       — Bengio & Senecal / Yi et al.: every denominator term,
        #                    the positive included, is corrected (= correct_positive);
        #   corrected      — Khrylchenko, Baikalov et al. (RecSys'25): the positive
        #                    is not sampled, so it leaves the denominator; negatives
        #                    use Q'(d) = q(d) / (1 - q(p)); the per-example loss is
        #                    weighted by sg(1 - P_hat(p|u)), P_hat estimated from the
        #                    same negatives with a 1/n mean in the denominator.
        if correct_positive:
            variant = 'standard'
        if variant not in ('negatives_only', 'standard', 'corrected'):
            raise ValueError(f'Unknown logQ variant `{variant}`')
        self._variant = variant
        self._correct_positive = variant == 'standard'
        if variant != 'negatives_only' and self._leave_own_out:
            raise ValueError('leave_own_out is only defined for the negatives_only variant')
        self._normalize_embeddings = normalize_embeddings
        self._temperature = temperature

        counts, _ = load_counts_artifact(
            path_to_item_counts,
            expected_role=expected_counts_role,
            expected_max_len=expected_counts_max_len,
        )

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
            expected_counts_role=config.get('expected_counts_role'),
            expected_counts_max_len=config.get('expected_counts_max_len'),
            correct_positive=config.get('correct_positive', False),
            variant=config.get('variant', 'negatives_only'),
            mixed_uniform_negatives=config.get('mixed_uniform_negatives', 0),
            table_prefix=config.get('table_prefix'),
        )

    def _mns_forward(self, inputs, queries, pos_embs, pos_ids):
        """In-batch pool + K shared uniform negatives; three variants over the same
        (B, B+K) score matrix with the mixture proposal in the correction."""
        table = inputs[self._table_prefix]  # (num_items + 2, D)
        num_items = table.shape[0] - 2
        batch_size = queries.size(0)
        K = self._mixed_uniform_negatives
        device = queries.device
        uniform_ids = torch.randint(1, num_items + 1, (K,), device=device)
        cand_ids = torch.cat((pos_ids, uniform_ids))  # (B + K,)
        cand_embs = torch.cat((pos_embs, table[uniform_ids]), dim=0)  # (B + K, D)
        if self._normalize_embeddings:
            queries = torch.nn.functional.normalize(queries, p=2, dim=-1, eps=1e-6)
            cand_embs = torch.nn.functional.normalize(cand_embs, p=2, dim=-1, eps=1e-6)
        scores = torch.mm(queries, cand_embs.T) / self._temperature  # (B, B + K)
        q = self._prob_table[cand_ids]
        share = batch_size / (batch_size + K)
        q_mix = torch.clamp(share * q + (1.0 - share) / num_items, min=1e-10)
        log_q = torch.log(q_mix)  # (B + K,)
        rows = torch.arange(batch_size, device=device)
        # masks: accidental hits (same item as the row's positive) and same-user rows
        invalid = cand_ids.unsqueeze(0) == pos_ids.unsqueeze(1)  # (B, B + K)
        invalid[rows, rows] = False
        if self._user_ids_prefix is not None:
            user_ids = inputs[self._user_ids_prefix].reshape(-1)
            same_user = user_ids.unsqueeze(0) == user_ids.unsqueeze(1)
            same_user.fill_diagonal_(False)
            invalid[:, :batch_size] |= same_user
        if self._variant == 'corrected':
            invalid_c = invalid.clone()
            invalid_c[rows, rows] = True
            pos_mix = q_mix[:batch_size]
            row_keep_log = torch.log(torch.clamp(1.0 - pos_mix, min=1e-10))
            negatives = (
                scores - self._logq_lambda * log_q.unsqueeze(0) + self._logq_lambda * row_keep_log.unsqueeze(1)
            ).masked_fill(invalid_c, -1e12)
            positive = scores.diagonal()
            log_sum_neg = torch.logsumexp(negatives, dim=1)
            per_example = log_sum_neg - positive
            num_valid = (~invalid_c).sum(dim=1).clamp(min=1).float()
            log_p_hat = positive - torch.logaddexp(positive, log_sum_neg - torch.log(num_valid))
            weight = (1.0 - torch.exp(log_p_hat)).detach()
            return (weight * per_example).mean()
        corrected = scores - self._logq_lambda * log_q.unsqueeze(0)
        if self._variant == 'negatives_only':
            corrected[rows, rows] += self._logq_lambda * log_q[:batch_size]  # positive uncorrected
        corrected = corrected.masked_fill(invalid, -1e12)
        return torch.nn.functional.cross_entropy(corrected, rows)

    def _corrected_forward(self, all_scores, pos_ids, invalid_mask):
        """RecSys'25 corrected logQ: L = -w * (s_pos - log sum_{neg} exp(s_neg - lambda log Q'(neg))),
        Q'_i(j) = q(j) / (1 - q(p_i)), w = sg(1 - P_hat), P_hat = e^{s_pos} / (e^{s_pos} + mean_neg e^{corrected})."""
        log_q = self._log_q_table[pos_ids]  # (B,)
        row_keep_log = torch.log(torch.clamp(1.0 - self._prob_table[pos_ids], min=1e-10))  # log(1 - q(p_i))
        negatives = (
            all_scores - self._logq_lambda * log_q.unsqueeze(0) + self._logq_lambda * row_keep_log.unsqueeze(1)
        ).masked_fill(invalid_mask, -1e12)  # (B, B), the diagonal is invalid: the positive leaves the denominator
        positive = all_scores.diagonal()  # (B,), uncorrected (a constant shift does not change gradients)
        log_sum_neg = torch.logsumexp(negatives, dim=1)  # (B,)
        per_example = log_sum_neg - positive  # -log(e^{s_pos} / sum_neg e^{corrected})
        num_valid = (~invalid_mask).sum(dim=1).clamp(min=1).float()
        log_p_hat = positive - torch.logaddexp(positive, log_sum_neg - torch.log(num_valid))
        weight = (1.0 - torch.exp(log_p_hat)).detach()
        return (weight * per_example).mean()

    def _invalid_mask(self, inputs, pos_ids):
        """(B, B) mask of in-batch candidates that must not serve as negatives for a row:
        another row whose positive is the same item (false negative) and, when user ids
        are given, another prefix of the same user (its own future item). Diagonal False."""
        invalid = pos_ids.unsqueeze(0) == pos_ids.unsqueeze(1)
        invalid.fill_diagonal_(False)
        if self._user_ids_prefix is not None:
            user_ids = inputs[self._user_ids_prefix].reshape(-1)  # (B,)
            same_user = user_ids.unsqueeze(0) == user_ids.unsqueeze(1)
            same_user.fill_diagonal_(False)
            invalid = invalid | same_user
        return invalid

    def _sampled_softmax_forward(self, scores, pos_ids, invalid):
        """negatives_only / standard forms. Every column is shifted by -lambda * log q(j);
        negatives_only then restores the diagonal (the positive is observed, not sampled),
        leave_own_out uses q'_i(j) = q(j) / (1 - q(p_i)) instead of q."""
        log_q = self._log_q_table[pos_ids]  # (B,)
        scores = scores - self._logq_lambda * log_q.unsqueeze(0)  # (B, B)
        if self._leave_own_out:
            row_keep_log = torch.log(torch.clamp(1.0 - self._prob_table[pos_ids], min=1e-10))  # log(1 - q(p_i))
            scores = scores + self._logq_lambda * row_keep_log.unsqueeze(1)
            scores.diagonal().add_(self._logq_lambda * (log_q - row_keep_log))
        elif not self._correct_positive:
            scores.diagonal().add_(self._logq_lambda * log_q)
        scores = scores.masked_fill(invalid, -1e12)
        labels = torch.arange(scores.size(0), device=scores.device)  # the positive sits on the diagonal
        return torch.nn.functional.cross_entropy(scores, labels)

    def forward(self, inputs):
        queries = inputs[self._queries_prefix]  # (B, D)
        pos_embs = inputs[self._positive_prefix]  # (B, D)
        pos_ids = inputs[self._positive_ids_prefix]  # (B,)

        if self._mixed_uniform_negatives > 0:
            loss = self._mns_forward(inputs, queries, pos_embs, pos_ids)
        else:
            if self._normalize_embeddings:
                queries = torch.nn.functional.normalize(queries, p=2, dim=-1, eps=1e-6)
                pos_embs = torch.nn.functional.normalize(pos_embs, p=2, dim=-1, eps=1e-6)
            # all-pairs scores: row i is query i against every positive in the batch;
            # the diagonal is the row's own positive, everything else is an in-batch negative
            scores = torch.mm(queries, pos_embs.T) / self._temperature  # (B, B)
            invalid = self._invalid_mask(inputs, pos_ids)
            if self._variant == 'corrected':
                eye = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
                loss = self._corrected_forward(scores, pos_ids, invalid | eye)
            else:
                loss = self._sampled_softmax_forward(scores, pos_ids, invalid)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()
        return loss


class FullSoftmaxLoss(TorchLoss, config_name='full_softmax'):
    """
    Exact full-catalog softmax cross-entropy for the retrieval loss.

    No negative sampling and hence no sampling bias: the denominator sums over
    the whole item embedding table (padding and mask columns excluded). The
    gold-standard anchor that in-batch + logQ approximates.
    """

    def __init__(
        self,
        queries_prefix,
        table_prefix,
        positive_ids_prefix,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._table_prefix = table_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            queries_prefix=config['queries_prefix'],
            table_prefix=config['table_prefix'],
            positive_ids_prefix=config['positive_ids_prefix'],
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        queries = inputs[self._queries_prefix]  # (B, D)
        table = inputs[self._table_prefix]  # (num_items + 2, D)
        pos_ids = inputs[self._positive_ids_prefix]  # (B,)

        all_scores = torch.mm(queries, table.T)  # (B, num_items + 2)
        all_scores[:, 0] = -1e12  # padding column
        all_scores[:, -1] = -1e12  # mask-token column

        loss = torch.nn.functional.cross_entropy(all_scores, pos_ids)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SampledLogQLoss(TorchLoss, config_name='mclsr_logq_special'):
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
        counts, _ = load_counts_artifact(path_to_item_counts)

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
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        # 1. Extract embeddings and item IDs
        queries = inputs[self._queries_prefix]  # (Batch, Dim)
        pos_embs = inputs[self._positive_prefix]  # (Batch, Dim)
        neg_embs = inputs[self._negative_prefix]  # (Batch, NumNegs, Dim)

        pos_ids = inputs[self._positive_ids_prefix]  # (Batch)
        neg_ids = inputs[self._negative_ids_prefix]  # (Batch, NumNegs)

        # 2. Compute raw scores (Dot Product)
        # Using einsum for efficient multiplication of 2D queries and 3D negatives
        pos_scores = torch.einsum('bd,bd->b', queries, pos_embs).unsqueeze(-1)  # (B, 1)
        neg_scores = torch.einsum('bd,bnd->bn', queries, neg_embs)  # (B, N)

        # 3. False Negative Masking
        # Neutralize cases where the sampled negative item is actually the target item
        false_negative_mask = pos_ids.unsqueeze(1) == neg_ids
        neg_scores = neg_scores.masked_fill(false_negative_mask, -1e12)

        # 4. Apply LogQ Correction
        # Correction term: score = score - lambda * log(p_j)
        log_q_pos = self._log_q_table[pos_ids].unsqueeze(-1)  # (B, 1)
        log_q_neg = self._log_q_table[neg_ids]  # (B, N)

        pos_scores = pos_scores - (self._logq_lambda * log_q_pos)
        neg_scores = neg_scores - (self._logq_lambda * log_q_neg)

        # 5. Final Softmax Reranking
        # Concatenate scores and compute cross-entropy over the sampled items
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # (B, 1+N)
        loss = -torch.log_softmax(all_scores, dim=1)[:, 0]

        final_loss = loss.mean()
        if self._output_prefix:
            inputs[self._output_prefix] = final_loss.cpu().item()

        return final_loss


class SampledSoftmaxLoss(TorchLoss, config_name='sampled_softmax'):
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
        queries_embeddings = inputs[self._queries_prefix]  # (batch_size, embedding_dim)
        positive_embeddings = inputs[self._positive_prefix]  # (batch_size, embedding_dim)
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
            assert negative_embeddings.dim() == 3  # (batch_size, num_negatives, embedding_dim)
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


class SASRecLoss(TorchLoss, config_name='sasrec'):
    def __init__(self, positive_prefix, negative_prefix, output_prefix=None):
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
        ) + torch.nn.functional.binary_cross_entropy_with_logits(negative_scores, torch.zeros_like(negative_scores))

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
