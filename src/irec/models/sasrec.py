from irec.models import SequentialTorchModel
from irec.utils import create_masked_tensor

import torch


class SasRecModel(SequentialTorchModel, config_name='sasrec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02,
            eval_top_k=50,
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._eval_top_k = eval_top_k

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-9),
            eval_top_k=config.get('eval_top_k', 50),
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

            all_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)

            # a -- all_batch_events, n -- num_items, d -- embedding_dim
            all_scores = torch.einsum(
                'ad,nd->an',
                all_sample_embeddings,
                all_embeddings
            )  # (all_batch_events, num_items)

            positive_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=all_positive_sample_events[..., None]
            )[:, 0]  # (all_batch_items)

            # rejection sampling over REAL items (1..num_items), excluding the
            # query's WHOLE consumed sequence and its target — the standard
            # SASRec negative protocol. (The historical BCE runs sampled the
            # full id range incl. padding/mask with no exclusions — footnoted
            # in RESULTS; those numbers are approximately reproducible only.)
            device = all_positive_sample_events.device
            sample_ids_padded, _ = create_masked_tensor(
                data=all_sample_events,
                lengths=all_sample_lengths,
            )  # (batch_size, seq_len); padded with 0 = never a real candidate
            # forbidden set per user = FULL sequence: inputs UNION positives
            # (the final target appears only among positives, so inputs alone
            # would still let early queries draw it as a negative)
            last_positive = all_positive_sample_events[
                torch.cumsum(all_sample_lengths, dim=0) - 1
            ]  # (batch_size,)
            per_user_forbidden = torch.cat(
                (sample_ids_padded, last_positive[:, None]), dim=1,
            )
            forbidden = torch.repeat_interleave(
                per_user_forbidden, all_sample_lengths, dim=0,
            )  # (all_batch_events, seq_len + 1), identical rows per user
            negative_ids = torch.randint(
                1, self._num_items + 1,
                size=all_positive_sample_events.shape, device=device,
            )
            bad = torch.ones_like(negative_ids, dtype=torch.bool)
            for _ in range(64):  # expected #collisions per pass ~ len/|V| << 1
                bad = (negative_ids[:, None] == forbidden).any(dim=1)
                if not bad.any():
                    break
                negative_ids[bad] = torch.randint(
                    1, self._num_items + 1,
                    size=(int(bad.sum()),), device=device,
                )
            if bad.any():
                raise RuntimeError(
                    'negative rejection sampling could not find eligible items '
                    'for {} queries — the catalog is too dense relative to '
                    'user sequences; use exact eligible-set sampling'.format(
                        int(bad.sum()),
                    )
                )
            negative_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=negative_ids[..., None]
            )[:, 0]  # (all_batch_items)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores,
                'negative_ids': negative_ids,  # exposed for tests/inspection
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf
            if self._eval_top_k > self._num_items:
                raise ValueError(
                    'eval_top_k={} is larger than num_items={}'.format(
                        self._eval_top_k, self._num_items,
                    ),
                )

            _, indices = torch.topk(
                candidate_scores,
                k=self._eval_top_k, dim=-1, largest=True
            )  # (batch_size, eval_top_k)

            return indices


class SasRecInBatchModel(SasRecModel, config_name='sasrec_in_batch'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02,
            eval_top_k=50,
    ):
        super().__init__(
            sequence_prefix=sequence_prefix,
            positive_prefix=positive_prefix,
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            eval_top_k=eval_top_k,
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-9),
            eval_top_k=config.get('eval_top_k', 50),
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (all_batch_events, embedding_dim)

            # in-batch training: negatives ARE the other positives in the batch,
            # so only queries/positives/ids are emitted (an explicit random
            # negative draw used to live here — unused by the loss, removed;
            # NB this changes the RNG stream relative to the historical runs)
            return {
                'query_embeddings': in_batch_queries_embeddings,
                'positive_embeddings': in_batch_positive_embeddings,
                # raw ids enable false-negative masking and per-item logQ
                'positive_ids': in_batch_positive_events,
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf
            if self._eval_top_k > self._num_items:
                raise ValueError(
                    'eval_top_k={} is larger than num_items={}'.format(
                        self._eval_top_k, self._num_items,
                    ),
                )

            _, indices = torch.topk(
                candidate_scores,
                k=self._eval_top_k, dim=-1, largest=True
            )  # (batch_size, eval_top_k)

            return indices