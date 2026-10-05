"""Ranking metrics over the top-k index lists a model returns in eval mode; every metric yields one value per user."""

from irec.utils import MetaParent

import torch


class BaseMetric(metaclass=MetaParent):
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class MCLSRNDCGMetric(BaseMetric, config_name='mclsr-ndcg'):
    """ndcg@k with the standard normalisation (ideal DCG over min(k, number of targets)); one value per user."""

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]
        assert predictions.shape[1] >= self._k
        predictions = predictions[:, : self._k]  # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids']  # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length']  # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        dcg_scores = []
        offset = 0
        for i in range(predictions.shape[0]):
            user_predictions = predictions[i]
            num_user_labels = int(labels_lengths[i].item())
            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels

            hits_mask = torch.isin(user_predictions, user_labels)  # (k,) -> True/False

            positions = torch.arange(2, self._k + 2, device=predictions.device)
            weights = 1 / torch.log2(positions.float())
            dcg = (hits_mask.float() * weights).sum()

            num_ideal_hits = min(self._k, num_user_labels)
            idcg_weights = weights[:num_ideal_hits]
            idcg = idcg_weights.sum()

            ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0)
            dcg_scores.append(ndcg.cpu().item())

        return dcg_scores


class ComiRecNDCGMetric(BaseMetric, config_name='comirec-ndcg'):
    """
    NDCG with the ComiRec normalization (used by the MCLSR paper, Table 2):
    IDCG is computed over the number of ACTUAL hits, not over the ideal
    min(k, num_targets). Yields values >= the standard NDCG; needed for
    apples-to-apples comparison with the paper's reported numbers.
    """

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]
        assert predictions.shape[1] >= self._k
        predictions = predictions[:, : self._k]
        labels_flat = inputs[f'{labels_prefix}.ids']
        labels_lengths = inputs[f'{labels_prefix}.length']

        assert predictions.shape[0] == labels_lengths.shape[0]

        scores = []
        offset = 0
        weights = 1 / torch.log2(
            torch.arange(2, self._k + 2, device=predictions.device).float(),
        )
        for i in range(predictions.shape[0]):
            num_user_labels = int(labels_lengths[i].item())
            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels

            hits_mask = torch.isin(predictions[i], user_labels)
            num_hits = int(hits_mask.sum().item())
            if num_hits == 0:
                scores.append(0.0)
                continue
            dcg = (hits_mask.float() * weights).sum()
            idcg = weights[:num_hits].sum()
            scores.append((dcg / idcg).cpu().item())

        return scores


class MCLSRRecallMetric(BaseMetric, config_name='mclsr-recall'):
    """recall@k = hits among the top k / number of targets; one value per user."""

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]
        assert predictions.shape[1] >= self._k
        predictions = predictions[:, : self._k]  # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids']  # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length']  # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        recall_scores = []
        offset = 0
        for i in range(predictions.shape[0]):
            user_predictions = predictions[i]
            num_user_labels = int(labels_lengths[i].item())
            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels

            hits = torch.isin(user_predictions, user_labels).sum().float()

            recall = hits / num_user_labels if num_user_labels > 0 else torch.tensor(0.0)
            recall_scores.append(recall.cpu().item())

        return recall_scores


class MCLSRHitRateMetric(BaseMetric, config_name='mclsr-hit'):
    """hit@k = 1 if any target is among the top k; one value per user."""

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]
        assert predictions.shape[1] >= self._k
        predictions = predictions[:, : self._k]  # (batch_size, k)
        labels_flat = inputs[f'{labels_prefix}.ids']  # (total_labels,)
        labels_lengths = inputs[f'{labels_prefix}.length']  # (batch_size,)

        assert predictions.shape[0] == labels_lengths.shape[0]

        hit_scores = []
        offset = 0
        for i in range(predictions.shape[0]):
            user_predictions = predictions[i]
            num_user_labels = int(labels_lengths[i].item())

            if num_user_labels == 0:
                hit_scores.append(0.0)
                continue

            user_labels = labels_flat[offset : offset + num_user_labels]
            offset += num_user_labels

            is_hit = torch.isin(user_predictions, user_labels).any()

            hit_scores.append(float(is_hit))

        return hit_scores
