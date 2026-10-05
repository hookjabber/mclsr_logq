"""Contrastive losses: the interest- and feature-level objectives of the paper (eq. 8, 11, 12).

* ``FpsLoss`` (``fps``): two-view contrastive loss; ``scheme`` = ``symmetric`` (SimCLR-style, the study
  default), ``paper`` (eq. 8 / 11 literally) or ``cross_only``; dot or euclidean scores, optional cosine;
* ``FpsLogQLoss`` (``fps_logq``): the logQ correction applied to a two-view contrastive loss;
* ``MatchedContrastiveFullSoftmaxLoss`` / ``ContrastiveFullSoftmaxLoss``: full-catalogue contrastive anchors.
"""

import os
import pickle
import torch
import torch.nn as nn

from .base import TorchLoss
from .counts import load_counts_artifact


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
                queries,
                candidates,
                compute_mode='donot_use_mm_for_euclid_dist',
            )
            return -(distances**2) / self._tau
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
            output_prefix=config.get('output_prefix'),
        )

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        if self._scheme in ('cross_only', 'paper'):
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
            combined_embeddings,
            combined_embeddings,
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

        labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()  # (2 * x)
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

        if self._scheme not in ('symmetric', 'cross_only', 'paper'):
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

        if self._scheme == 'paper':
            # paper eq. 8 / 11: anchors = fst view only; candidates = every snd row plus the other fst
            # rows = the first B rows of the symmetric table with the own-copy diagonal masked;
            # same corrections and false-negative masks, mean over B anchors (no /2)
            loss = self._loss_function(all_scores[:batch_size], positive_indices[:batch_size])
        else:
            loss = self._loss_function(all_scores, positive_indices) / 2

        if self._output_prefix is not None:
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
