"""Per-user ranking metrics with binary relevance over unique target items.

Repeated target events do not create additional relevant catalogue items. Ranked
predictions must be distinct positive IDs; padding is not an evaluation target.
"""

from irec.utils import MetaParent

import torch


class BaseMetric(metaclass=MetaParent):
    pass


class StatefullMetric(BaseMetric):
    def reduce(self):
        raise NotImplementedError


class _RankingMetric(BaseMetric):
    def __init__(self, k):
        if type(k) is not int or k < 1:
            raise ValueError('metric k must be a positive integer')
        self._k = k

    def _rows(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix]
        labels = inputs[f'{labels_prefix}.ids']
        lengths = inputs[f'{labels_prefix}.length']
        if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in (predictions, labels, lengths)):
            raise ValueError('predictions, target IDs and target lengths must be integer tensors')
        if predictions.ndim != 2 or predictions.shape[1] < self._k:
            raise ValueError('predictions must contain at least k ranked items per user')
        if labels.ndim != 1 or lengths.ndim != 1 or predictions.shape[0] != lengths.shape[0]:
            raise ValueError('flat target IDs and per-user lengths must match the prediction batch')
        if (lengths < 0).any() or int(lengths.sum()) != labels.numel():
            raise ValueError('target lengths must be nonnegative and account for every target ID')
        if (labels <= 0).any() or (predictions[:, : self._k] <= 0).any():
            raise ValueError('ranking and target IDs must be positive, without padding')
        offset = 0
        for row, length in zip(predictions[:, : self._k], lengths.tolist()):
            if torch.unique(row).numel() != self._k:
                raise ValueError('ranked predictions must contain unique item IDs')
            relevant = torch.unique(labels[offset : offset + length])
            offset += length
            yield row, relevant


class MCLSRNDCGMetric(_RankingMetric, config_name='mclsr-ndcg'):
    """Standard binary-relevance NDCG; ideal DCG uses min(k, unique targets)."""

    def __call__(self, inputs, pred_prefix, labels_prefix):
        scores = []
        for predictions, relevant in self._rows(inputs, pred_prefix, labels_prefix):
            weights = 1 / torch.log2(torch.arange(2, self._k + 2, device=predictions.device).float())
            hits = torch.isin(predictions, relevant)
            dcg = (hits.float() * weights).sum()
            ideal = weights[: min(self._k, relevant.numel())].sum()
            scores.append((dcg / ideal).item() if relevant.numel() else 0.0)
        return scores


class ComiRecNDCGMetric(_RankingMetric, config_name='comirec-ndcg'):
    """Historical alternative: normalize DCG by actual hits, not all relevant items.

    This differs from standard NDCG. Exact equivalence to the original paper's
    evaluator has not been established from the available author artifacts.
    """

    def __call__(self, inputs, pred_prefix, labels_prefix):
        scores = []
        for predictions, relevant in self._rows(inputs, pred_prefix, labels_prefix):
            hits = torch.isin(predictions, relevant)
            num_hits = int(hits.sum())
            if not num_hits:
                scores.append(0.0)
                continue
            weights = 1 / torch.log2(torch.arange(2, self._k + 2, device=predictions.device).float())
            dcg = (hits.float() * weights).sum()
            scores.append((dcg / weights[:num_hits].sum()).item())
        return scores


class MCLSRRecallMetric(_RankingMetric, config_name='mclsr-recall'):
    """Recall@k = unique relevant items retrieved / unique target items."""

    def __call__(self, inputs, pred_prefix, labels_prefix):
        scores = []
        for predictions, relevant in self._rows(inputs, pred_prefix, labels_prefix):
            hits = torch.isin(predictions, relevant).sum().float()
            scores.append((hits / relevant.numel()).item() if relevant.numel() else 0.0)
        return scores


class MCLSRHitRateMetric(_RankingMetric, config_name='mclsr-hit'):
    """Hit@k = 1 if any relevant item appears in the ranking; one value per user."""

    def __call__(self, inputs, pred_prefix, labels_prefix):
        return [
            float(torch.isin(predictions, relevant).any())
            for predictions, relevant in self._rows(inputs, pred_prefix, labels_prefix)
        ]
