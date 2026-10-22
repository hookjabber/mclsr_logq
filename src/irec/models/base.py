"""Model base classes: weight initialisation, the last-position gather and the causal transformer encoder shared by the SASRec models."""

from irec.utils import MetaParent

from irec.utils import (
    DEVICE,
    create_masked_tensor,
    get_activation_function,
    create_logger,
)

import torch
import torch.nn as nn

logger = create_logger(name=__name__)


class BaseModel(metaclass=MetaParent):
    pass


class TorchModel(nn.Module, BaseModel):
    """Registry root of the torch models: shared weight initialisation and the last-position gather."""

    @staticmethod
    def _validate_sequence_inputs(events, lengths):
        if lengths.numel() == 0 or (lengths <= 0).any():
            raise ValueError('non-empty histories require positive sequence lengths')
        if lengths.sum().item() != events.numel():
            raise ValueError('sequence lengths must sum to the number of events')

    def _rank_catalogue(self, scores):
        if type(self._eval_top_k) is not int or not 1 <= self._eval_top_k <= self._num_items:
            raise ValueError(f'eval_top_k must be a positive integer at most num_items={self._num_items}')
        if not torch.isfinite(scores[:, 1 : self._num_items + 1]).all():
            raise ValueError('catalogue ranking requires finite scores for every real item')
        scores[:, 0] = -torch.inf
        scores[:, self._num_items + 1 :] = -torch.inf
        return torch.topk(scores, k=self._eval_top_k, dim=-1, largest=True).indices

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    @staticmethod
    def _get_last_embedding(embeddings, mask):
        lengths = torch.sum(mask, dim=-1)  # (batch_size)
        lengths = lengths - 1  # (batch_size)
        last_masks = mask.gather(
            dim=1,
            index=lengths[:, None],
        )  # (batch_size, 1)
        lengths = torch.tile(
            lengths[:, None, None],
            (1, 1, embeddings.shape[-1]),
        )  # (batch_size, 1, emb_dim)
        last_embeddings = embeddings.gather(
            dim=1,
            index=lengths,
        )  # (batch_size, 1, emb_dim)
        last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)
        if not torch.allclose(embeddings[mask][-1], last_embeddings[-1]):
            logger.debug(f'Embeddings: {embeddings}')
            logger.debug(
                f'Lengths: {lengths}, max: {lengths.max()}, min: {lengths.min()}',
            )
            logger.debug(f'Last embedding from mask: {embeddings[mask][-1]}')
            logger.debug(f'Last embedding from gather: {last_embeddings[-1]}')
            assert False
        return last_embeddings


class SequentialTorchModel(TorchModel):
    """Causal transformer encoder over item + position embeddings (positions counted from the end), LayerNorm and dropout on the input, padding masked; the SASRec models build on it."""

    def __init__(
        self,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.0,
        activation='relu',
        layer_norm_eps=1e-5,
        is_causal=True,
    ):
        super().__init__()
        self._is_causal = is_causal
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim,
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim,
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers,
        )

    def _apply_sequential_encoder(self, events, lengths):
        self._validate_sequence_inputs(events, lengths)
        embeddings = self._item_embeddings(
            events,
        )  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths,
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(
                start=seq_len - 1,
                end=-1,
                step=-1,
                device=mask.device,
            )[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(
            positions,
        )  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=lengths,
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(
            embeddings,
        )  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(
            embeddings,
        )  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        if self._is_causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(DEVICE)  # (seq_len, seq_len)
            embeddings = self._encoder(
                src=embeddings,
                mask=~causal_mask,
                src_key_padding_mask=~mask,
            )  # (batch_size, seq_len, embedding_dim)
        else:
            embeddings = self._encoder(
                src=embeddings,
                src_key_padding_mask=~mask,
            )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask
