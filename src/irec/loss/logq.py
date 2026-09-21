"""Losses of the logQ study.

The sampled-softmax family and every form of the logQ sampling-bias correction used in the
study live here; the framework losses (composite, FPS contrastive, SASRec, sampled softmax with
explicit negatives) stay in ``base.py``.

* ``MCLSRLogqInBatchLoss`` (``mclsr_logq_inbatch``) -- in-batch sampled softmax for the retrieval
  loss L_P with three forms of the correction: ``negatives_only`` (the study default), ``standard``
  (Bengio & Senecal; Yi et al.) and ``corrected`` (Khrylchenko, Baikalov et al., RecSys'25), plus
  mixed negative sampling (in-batch pool + shared uniform negatives, mixture proposal).
* ``FullSoftmaxLoss`` (``full_softmax``) -- the exact full-catalogue softmax, the no-sampling anchor.
* ``FpsLogQLoss`` (``fps_logq``) -- the correction on a two-view contrastive loss (L_IL, L_UC, L_IC).
* ``MCLSRLogqLoss`` (``mclsr_logq_special``) -- the correction with explicit sampled negatives.
* ``MatchedContrastiveFullSoftmaxLoss`` / ``ContrastiveFullSoftmaxLoss`` -- full-catalogue
  contrastive anchors.

Count tables are loaded through ``load_counts_artifact``; a table carries its role (target /
context-inclusion) and provenance, and a config that states expectations is checked against them.
"""

import os
import pickle

import torch
import torch.nn as nn

from .base import TorchLoss


def load_counts_artifact(path, expected_role=None, expected_max_len=None):
    """Load a counts table: either a bare array (legacy) or a metadata dict
    {'counts', 'denominator', 'role', 'max_len', 'source', 'source_sha256'}.
    Returns (counts_array, metadata_dict); metadata is {} for bare arrays.

    When the config states expectations, they are ENFORCED: a table with the
    wrong role/max_len (e.g. target counts wired where context-inclusion Q
    belongs — both may share a denominator by coincidence) fails loudly, and
    expectations against a legacy bare array fail too (it cannot prove them)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Counts file not found at {path}')
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if isinstance(artifact, dict):
        metadata = {k: v for k, v in artifact.items() if k != 'counts'}
        counts = artifact['counts']
    else:
        metadata, counts = {}, artifact
    if expected_role is not None and metadata.get('role') != expected_role:
        raise ValueError(
            'counts artifact role mismatch at {}: expected {!r}, got {!r}'.format(
                path,
                expected_role,
                metadata.get('role'),
            )
        )
    if expected_max_len is not None and metadata.get('max_len') != expected_max_len:
        raise ValueError(
            'counts artifact max_len mismatch at {}: expected {}, got {}'.format(
                path,
                expected_max_len,
                metadata.get('max_len'),
            )
        )
    return counts, metadata


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
        num_draws_value=None,
        counts_denominator=None,
        expected_counts_role=None,
        expected_counts_max_len=None,
        similarity='dot',
        center_log_q=False,
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._ids_prefix = ids_prefix
        self._tau = tau
        self._similarity = similarity
        # keep only the zero-mean part of the correction: removes the uniform
        # negatives-vs-positive margin, leaves pure reweighting between candidates
        self._center_log_q = center_log_q
        if self._similarity not in ('dot', 'euclidean'):
            raise ValueError(f'Unknown fps_logq similarity `{self._similarity}`')
        self._logq_lambda = logq_lambda
        self._logq_probability_mode = logq_probability_mode
        self._num_draws_prefix = num_draws_prefix
        self._num_draws_value = num_draws_value
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

        counts, artifact_meta = load_counts_artifact(
            path_to_counts,
            expected_role=expected_counts_role,
            expected_max_len=expected_counts_max_len,
        )

        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        # counts_denominator overrides the sum for tables whose natural unit is
        # not an event share — e.g. line-inclusion counts, where q(v) must be
        # count(v) / num_lines (probability a random LINE includes v), not
        # count(v) / sum-of-all-inclusions. A metadata-carrying artifact states
        # its own denominator; an explicit config value must then AGREE with it
        # (guards against silently pairing a stale number with new data).
        artifact_denominator = artifact_meta.get('denominator')
        if (
            counts_denominator is not None
            and artifact_denominator is not None
            and float(counts_denominator) != float(artifact_denominator)
        ):
            raise ValueError(
                'counts_denominator={} disagrees with the artifact metadata denominator={} ({})'.format(
                    counts_denominator,
                    artifact_denominator,
                    path_to_counts,
                )
            )
        if counts_denominator is None:
            counts_denominator = artifact_denominator
        denominator = counts_tensor.sum() if counts_denominator is None else float(counts_denominator)
        probs = torch.clamp(counts_tensor / denominator, min=1e-10, max=1.0)
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
            raise ValueError('FpsLogQLoss requires `path_to_counts` in the loss config')

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
            num_draws_value=config.get('num_draws_value'),
            counts_denominator=config.get('counts_denominator'),
            expected_counts_role=config.get('expected_counts_role'),
            expected_counts_max_len=config.get('expected_counts_max_len'),
            similarity=config.get('similarity', 'dot'),
            center_log_q=config.get('center_log_q', False),
        )

    def _centered(self, candidate_log_q):
        if not self._center_log_q:
            return candidate_log_q
        if candidate_log_q.dim() == 2:
            return candidate_log_q - candidate_log_q.mean(dim=1, keepdim=True)
        return candidate_log_q - candidate_log_q.mean()

    def _pairwise_scores(self, queries, candidates):
        # same scoring as FpsLoss: dot / tau, or -||a-b||^2 / tau (exact
        # cdist mode keeps the symmetric scheme's mirrored diagonal exact)
        if self._similarity == 'euclidean':
            distances = torch.cdist(
                queries,
                candidates,
                compute_mode='donot_use_mm_for_euclid_dist',
            )
            return -(distances**2) / self._tau
        return torch.mm(queries, candidates.T) / self._tau

    def _sample_log_q(self, ids, num_negative_draws):
        if self._logq_probability_mode == 'sample':
            return self._log_q_table[ids]  # (B,)
        sample_probs = self._prob_table[ids]
        if num_negative_draws <= 0:
            sample_q = torch.full_like(sample_probs, 1e-10)
        else:
            sample_q = 1.0 - torch.pow(1.0 - sample_probs, num_negative_draws)
            sample_q = torch.clamp(sample_q, min=1e-10)
        return torch.log(sample_q)  # (B,)

    def _candidate_log_q(self, ids, num_negative_draws):
        sample_log_q = self._sample_log_q(ids, num_negative_draws)
        return torch.cat(
            (sample_log_q, sample_log_q),
            dim=0,
        )  # (2 * B,)

    def _loo_log_q_square(self, ids, num_negative_draws):
        # (B, B): row = anchor, column = candidate; q'_a(j) = p_j / (1 - p_a)
        sample_probs = self._prob_table[ids]  # (B,)
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

    def _candidate_log_q_leave_own_out(self, ids, num_negative_draws):
        # q'_anchor(v) = count(v) / (S - count(anchor)) = p(v) / (1 - p(anchor)):
        # the anchor's own id is masked out of the pool, every draw comes from q'.
        sample_probs = self._prob_table[ids]  # (B,)
        candidate_probs = torch.cat((sample_probs, sample_probs))  # (2 * B,)
        anchor_keep = torch.clamp(1.0 - candidate_probs, min=1e-10)  # (2 * B,)
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
        ids = inputs[self._ids_prefix]  # (B,)

        if fst_embeddings.shape != snd_embeddings.shape:
            raise ValueError('FpsLogQLoss expects both embedding tensors to have the same shape')

        batch_size = fst_embeddings.shape[0]
        if ids.shape[0] != batch_size:
            raise ValueError(f'FpsLogQLoss got {ids.shape[0]} ids for batch size {batch_size}')

        if self._scheme == 'cross_only':
            # B x B: anchors = fst view, candidates = snd view only
            # (one negative per other sample, not two)
            if self._normalize_embeddings:
                fst_embeddings = torch.nn.functional.normalize(
                    fst_embeddings,
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )
                snd_embeddings = torch.nn.functional.normalize(
                    snd_embeddings,
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )
            all_scores = self._pairwise_scores(fst_embeddings, snd_embeddings)
            device = all_scores.device
            ids = ids.to(device=device)
            if self._num_draws_prefix is not None:
                num_negative_draws = int(inputs[self._num_draws_prefix])
            elif self._num_draws_value is not None:
                num_negative_draws = int(self._num_draws_value)
            else:
                num_negative_draws = batch_size - 1

            if self._leave_own_out:
                candidate_log_q = self._centered(
                    self._loo_log_q_square(
                        ids,
                        num_negative_draws,
                    )
                )  # (B, B)
                all_scores = all_scores - self._logq_lambda * candidate_log_q
                positive_log_q = candidate_log_q.diagonal()
            else:
                candidate_log_q = self._centered(
                    self._sample_log_q(
                        ids,
                        num_negative_draws,
                    )
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

        all_scores = self._pairwise_scores(
            combined_embeddings,
            combined_embeddings,
        )  # (2 * B, 2 * B)

        device = all_scores.device
        ids = ids.to(device=device)
        if self._num_draws_prefix is not None:
            num_negative_draws = int(inputs[self._num_draws_prefix])
        elif self._num_draws_value is not None:
            num_negative_draws = int(self._num_draws_value)
        else:
            num_negative_draws = batch_size - 1

        if self._leave_own_out:
            candidate_log_q = self._centered(
                self._candidate_log_q_leave_own_out(
                    ids,
                    num_negative_draws,
                )
            )  # (2 * B, 2 * B), row = anchor, column = candidate
            all_scores = all_scores - self._logq_lambda * candidate_log_q
        else:
            candidate_log_q = self._centered(
                self._candidate_log_q(
                    ids,
                    num_negative_draws,
                )
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
        all_scores[row_indices, positive_indices] += self._logq_lambda * positive_log_q

        invalid_mask = torch.eye(
            num_views,
            dtype=torch.bool,
            device=device,
        )
        if self._mask_false_negatives:
            candidate_ids = torch.cat((ids, ids), dim=0)  # (2 * B,)
            false_negative_mask = candidate_ids.unsqueeze(0) == candidate_ids.unsqueeze(1)
            false_negative_mask[row_indices, positive_indices] = False
            false_negative_mask.fill_diagonal_(False)
            invalid_mask |= false_negative_mask

        all_scores = all_scores.masked_fill(invalid_mask, -1e12)

        loss = self._loss_function(all_scores, positive_indices) / 2

        if self._output_prefix is not None:
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


class MatchedContrastiveFullSoftmaxLoss(
    TorchLoss,
    config_name='contrastive_full_softmax_matched',
):
    """
    Full-catalog analog of the SYMMETRIC in-batch FpsLoss (matched objective).

    Anchors = both batch views (2B rows, like FpsLoss). For each anchor the
    candidates are the FULL other-view table plus the FULL same-view table with
    the anchor's own same-view row masked; the positive is the anchor's own row
    in the other-view table. Entities absent from the training split (per the
    explicit train-presence mask, which also excludes padding and mask ids) are
    masked out of both tables — NOT a count threshold: real train singletons
    (count == 1) stay valid, otherwise their positives would be masked whenever
    they appear as anchors. Reduction: mean cross-entropy over the 2B anchors,
    divided by 2 — the same convention as FpsLoss, so the two losses differ
    ONLY in the candidate pool (full catalog vs in-batch).
    """

    def __init__(
        self,
        fst_anchors_prefix,
        snd_anchors_prefix,
        fst_table_prefix,
        snd_table_prefix,
        ids_prefix,
        path_to_train_presence,
        tau=1.0,
        use_mean=True,
        output_prefix=None,
    ):
        super().__init__()
        self._fst_anchors_prefix = fst_anchors_prefix
        self._snd_anchors_prefix = snd_anchors_prefix
        self._fst_table_prefix = fst_table_prefix
        self._snd_table_prefix = snd_table_prefix
        self._ids_prefix = ids_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(
            reduction='mean' if use_mean else 'sum',
        )
        self._output_prefix = output_prefix

        if not os.path.exists(path_to_train_presence):
            raise FileNotFoundError(f'Train presence mask not found at {path_to_train_presence}')
        with open(path_to_train_presence, 'rb') as f:
            presence = pickle.load(f)
        self.register_buffer(
            '_invalid_columns',
            ~torch.tensor(presence, dtype=torch.bool),
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_anchors_prefix=config['fst_anchors_prefix'],
            snd_anchors_prefix=config['snd_anchors_prefix'],
            fst_table_prefix=config['fst_table_prefix'],
            snd_table_prefix=config['snd_table_prefix'],
            ids_prefix=config['ids_prefix'],
            path_to_train_presence=config['path_to_train_presence'],
            tau=config.get('temperature', 1.0),
            use_mean=config.get('use_mean', True),
            output_prefix=config.get('output_prefix'),
        )

    def _direction_scores(self, anchors, other_table, same_table, ids, invalid):
        # (B, 2N): [other-view table || same-view table]
        scores = (
            torch.cat(
                (
                    torch.mm(anchors, other_table.T),
                    torch.mm(anchors, same_table.T),
                ),
                dim=1,
            )
            / self._tau
        )
        n = other_table.shape[0]
        scores = scores.masked_fill(
            torch.cat((invalid, invalid), dim=0).unsqueeze(0),
            -1e12,
        )
        rows = torch.arange(anchors.shape[0], device=scores.device)
        scores[rows, n + ids] = -1e12  # the anchor's own same-view row
        return scores  # positive = column `ids` (other-view block)

    def forward(self, inputs):
        fst_anchors = inputs[self._fst_anchors_prefix]  # (B, D)
        snd_anchors = inputs[self._snd_anchors_prefix]  # (B, D)
        fst_table = inputs[self._fst_table_prefix]  # (N, D)
        snd_table = inputs[self._snd_table_prefix]  # (N, D)
        ids = inputs[self._ids_prefix].reshape(-1)  # (B,)

        invalid = self._invalid_columns.to(fst_table.device)
        if bool(invalid[ids].any()):
            raise ValueError(
                'matched full-softmax: batch anchor ids fall outside the '
                'train-presence mask — their positives would be masked out; '
                'the presence file does not match the training data',
            )

        scores_fst = self._direction_scores(
            fst_anchors,
            snd_table,
            fst_table,
            ids,
            invalid,
        )
        scores_snd = self._direction_scores(
            snd_anchors,
            fst_table,
            snd_table,
            ids,
            invalid,
        )
        all_scores = torch.cat((scores_fst, scores_snd), dim=0)  # (2B, 2N)
        labels = torch.cat((ids, ids), dim=0)

        loss = self._loss_function(all_scores, labels) / 2

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
        anchors = inputs[self._anchors_prefix]  # (B, D)
        table = inputs[self._table_prefix]  # (N, D)
        ids = inputs[self._ids_prefix].reshape(-1)  # (B,)

        if self._normalize_embeddings:
            anchors = torch.nn.functional.normalize(
                anchors,
                p=2,
                dim=-1,
                eps=1e-6,
            )
            table = torch.nn.functional.normalize(
                table,
                p=2,
                dim=-1,
                eps=1e-6,
            )

        all_scores = torch.mm(anchors, table.T) / self._tau  # (B, N)
        all_scores[:, 0] = -1e12  # padding column
        all_scores[:, -1] = -1e12  # mask-token column

        loss = self._loss_function(all_scores, ids)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
