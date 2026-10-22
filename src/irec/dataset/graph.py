"""The three graphs of MCLSR (paper §3.2) built from training interactions.

User-item bipartite adjacency with the symmetric normalisation of eq. 1; user-user and item-item
co-action graphs (R Rᵀ and Rᵀ R with the diagonal removed) restricted to the 50 strongest
neighbours of every node, normalised the same way. Graphs are cached as ``.npz`` next to the data;
``scripts/verify_graphs.py`` verifies the historical binary/legacy mode only.

``interaction_weighting='binary'`` preserves the study's presence-based graphs.
``'event_count'`` uses canonical windowed-v2 events, not occurrences in overlapping
training rows, for R and therefore for both co-action products. It is train-only.

Cache filenames remain compatible with the verification scripts. Each archive includes a
versioned manifest of its interaction content, options and dimensions and is replaced atomically.
Legacy archives without this provenance are rebuilt.
"""

from collections import Counter, defaultdict
from numbers import Integral
import hashlib
import json
import tempfile
import zipfile
import zlib
from tqdm import tqdm
from irec.utils import DEVICE
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import os
import logging
from .base import BaseDataset

logger = logging.getLogger(__name__)
_GRAPH_BUILDER_VERSION = 2


class GraphDataset(BaseDataset, config_name='graph'):
    """Wraps a sequence dataset and builds the user-item, user-user and item-item graphs from its training interactions (see the module docstring); exposes them through ``meta`` for the model."""

    def __init__(
        self,
        dataset,
        graph_dir_path,
        use_train_data_only=True,
        use_user_graph=False,
        use_item_graph=False,
        neighborhood_size=None,
        interaction_weighting='binary',
    ):
        if interaction_weighting not in ('binary', 'event_count'):
            raise ValueError("interaction_weighting must be 'binary' or 'event_count'")
        if interaction_weighting == 'event_count' and not use_train_data_only:
            raise ValueError('event_count graphs require train-only data')
        if neighborhood_size is not None and (
            not isinstance(neighborhood_size, Integral) or isinstance(neighborhood_size, bool) or neighborhood_size < 1
        ):
            raise ValueError('neighborhood_size must be a positive integer or None')
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph
        self._neighborhood_size = neighborhood_size
        self._interaction_weighting = interaction_weighting
        self._weight_suffix = '' if interaction_weighting == 'binary' else '_event_count'

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

        canonical_events = None
        if interaction_weighting == 'event_count' or getattr(dataset, '_train_event_source', None) is not None:
            event_source = getattr(dataset, 'get_train_event_sequences', None)
            if not callable(event_source):
                raise ValueError('canonical windowed-v2 MCLSR event sequences are required')
            canonical_events = event_source()

        interactions_data = self._collect_interactions(train_sampler, validation_sampler, test_sampler)
        train_interactions = interactions_data['train_interactions']
        train_user_interactions = interactions_data['train_user_interactions']
        train_item_interactions = interactions_data['train_item_interactions']
        train_user_2_items = interactions_data['train_user_2_items']
        train_item_2_users = interactions_data['train_item_2_users']

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        self._interaction_counts = Counter(dict.fromkeys(train_interactions, 1))
        if interaction_weighting == 'event_count':
            self._interaction_counts = Counter(
                (user_id, item_id) for user_id, items in canonical_events for item_id in items
            )
            if self._interaction_counts.keys() != set(train_interactions):
                raise ValueError('canonical event support differs from training window support')

        cache_interactions = np.asarray(train_interactions, dtype='<i8').reshape(-1, 2)
        interaction_order = np.lexsort((cache_interactions[:, 1], cache_interactions[:, 0]))
        if interaction_weighting == 'event_count':
            weights = np.asarray([self._interaction_counts[pair] for pair in train_interactions], dtype='<i8')
            cache_interactions = np.column_stack((cache_interactions, weights))
        self._interactions_digest = hashlib.sha256(cache_interactions[interaction_order]).hexdigest()

        self._graph = self._build_or_load_bipartite_graph(
            graph_dir_path, train_user_interactions, train_item_interactions
        )

        self._user_graph = (
            self._build_or_load_similarity_graph(
                'user',
                self._train_user_interactions,
                self._train_item_interactions,
                train_item_2_users,
                train_user_2_items,
            )
            if self._use_user_graph
            else None
        )

        self._item_graph = (
            self._build_or_load_similarity_graph(
                'item',
                self._train_user_interactions,
                self._train_item_interactions,
                train_item_2_users,
                train_user_2_items,
            )
            if self._use_item_graph
            else None
        )

    def _build_or_load_similarity_graph(
        self, entity_type, train_user_interactions, train_item_interactions, train_item_2_users, train_user_2_items
    ):
        if entity_type not in ['user', 'item']:
            raise ValueError("entity_type must be either 'user' or 'item'")
        # have to delete and replace to not delete npz each time manually
        # path_to_graph = os.path.join(self._graph_dir_path, '{}_graph.npz'.format(entity_type))

        # instead better use such construction

        # neighborhood_size
        # The neighborhood_size is a filter that constrains the number of edges for each user or
        # item node in the graph.
        # k=50 implies that for each user, we find all possible neighbors, sort them based on
        # co-occurrence counts, and keep only the top 50. All other connections are removed from the graph.
        k_suffix = f'k{self._neighborhood_size}' if self._neighborhood_size is not None else 'full'
        train_suffix = 'trainOnly' if self._use_train_data_only else 'withValTest'
        filename = f'{entity_type}_graph_{k_suffix}{self._weight_suffix}_{train_suffix}.npz'
        path_to_graph = os.path.join(self._graph_dir_path, filename)

        is_user_graph = entity_type == 'user'
        num_entities = self._num_users if is_user_graph else self._num_items

        shape = (num_entities + 2, num_entities + 2)
        manifest = self._graph_cache_manifest(entity_type, shape)
        graph_matrix = self._load_cached_graph(path_to_graph, manifest, shape)
        if graph_matrix is None:
            interactions_fst = []
            interactions_snd = []
            interaction_weights = []
            visited_user_item_pairs = set()
            # have to delete cause
            # 3.2 Graph Construction
            # User-user/item-item graph
            # ..the weight of each edge denotes the number of co-action behaviors between user i and user j

            # visited_entity_pairs = set()

            for user_id, item_id in tqdm(
                zip(train_user_interactions, train_item_interactions),
                desc='Building {}-{} graph'.format(entity_type, entity_type),  # TODO need?
            ):
                if (user_id, item_id) in visited_user_item_pairs:
                    continue
                visited_user_item_pairs.add((user_id, item_id))

                # TODO look here at review
                source_entity = user_id if is_user_graph else item_id
                connection_map = train_item_2_users if is_user_graph else train_user_2_items
                connection_point = item_id if is_user_graph else user_id

                for connected_entity in connection_map[connection_point]:
                    if source_entity == connected_entity:
                        continue

                    # visited_entity_pairs.add(pair_key)
                    interactions_fst.append(source_entity)
                    interactions_snd.append(connected_entity)
                    other_pair = (connected_entity, item_id) if is_user_graph else (user_id, connected_entity)
                    interaction_weights.append(
                        self._interaction_counts[(user_id, item_id)] * self._interaction_counts[other_pair]
                    )

            connections = csr_matrix(
                (np.asarray(interaction_weights, dtype=float), (interactions_fst, interactions_snd)),
                shape=(num_entities + 2, num_entities + 2),
            )

            if self._neighborhood_size is not None:
                connections = self._filter_matrix_by_top_k(connections, self._neighborhood_size)

            graph_matrix = self.get_sparse_graph_layer(connections, num_entities + 2, num_entities + 2, biparite=False)
            self._save_cached_graph(path_to_graph, graph_matrix, manifest)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _build_or_load_bipartite_graph(self, graph_dir_path, train_user_interactions, train_item_interactions):
        # path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        train_suffix = 'trainOnly' if self._use_train_data_only else 'withValTest'
        filename = f'general_graph{self._weight_suffix}_{train_suffix}.npz'
        path_to_graph = os.path.join(graph_dir_path, filename)

        num_nodes = self._num_users + self._num_items + 4
        shape = (num_nodes, num_nodes)
        manifest = self._graph_cache_manifest('general', shape)
        graph_matrix = self._load_cached_graph(path_to_graph, manifest, shape)
        if graph_matrix is None:
            # place ones only when co-occurrence happens
            user2item_connections = csr_matrix(
                (
                    np.asarray(
                        [
                            self._interaction_counts[pair]
                            for pair in zip(train_user_interactions, train_item_interactions)
                        ],
                        dtype=float,
                    ),
                    (train_user_interactions, train_item_interactions),
                ),
                shape=(self._num_users + 2, self._num_items + 2),
            )  # (num_users + 2, num_items + 2), bipartite graph
            graph_matrix = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True,
            )
            self._save_cached_graph(path_to_graph, graph_matrix, manifest)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _graph_cache_manifest(self, entity_type, shape):
        """Bump _GRAPH_BUILDER_VERSION whenever graph semantics or cache serialization change."""
        return json.dumps(
            {
                'builder_version': _GRAPH_BUILDER_VERSION,
                'interaction_weighting': self._interaction_weighting,
                'interactions_sha256': self._interactions_digest,
                'entity_type': entity_type,
                'num_users': int(self._num_users),
                'num_items': int(self._num_items),
                'shape': shape,
                'use_train_data_only': bool(self._use_train_data_only),
                'neighborhood_size': (
                    int(self._neighborhood_size)
                    if entity_type != 'general' and self._neighborhood_size is not None
                    else None
                ),
            },
            sort_keys=True,
            separators=(',', ':'),
        )

    @staticmethod
    def _load_cached_graph(path_to_graph, manifest, shape):
        try:
            with np.load(path_to_graph, allow_pickle=False) as cached:
                if cached['cache_manifest'].item() != manifest:
                    logger.info('Rebuilding graph cache with changed provenance: %s', path_to_graph)
                    return None
                if cached['format'].item() != b'csr' or tuple(cached['shape']) != shape:
                    raise ValueError('Unexpected cached graph format or shape')
                graph_matrix = csr_matrix((cached['data'], cached['indices'], cached['indptr']), shape=shape)
                graph_matrix.check_format(full_check=True)
                if not np.isfinite(graph_matrix.data).all():
                    raise ValueError('Non-finite cached graph weights')
                return graph_matrix
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            logger.warning('Rebuilding untrusted graph cache %s: %s', path_to_graph, error)
            return None

    @staticmethod
    def _save_cached_graph(path_to_graph, graph_matrix, manifest):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(path_to_graph),
                prefix=f'.{os.path.basename(path_to_graph)}.',
                suffix='.tmp',
                delete=False,
            ) as cache_file:
                temporary_path = cache_file.name
                np.savez_compressed(
                    cache_file,
                    format=np.array(b'csr'),
                    shape=graph_matrix.shape,
                    data=graph_matrix.data,
                    indices=graph_matrix.indices,
                    indptr=graph_matrix.indptr,
                    cache_manifest=np.array(manifest),
                )
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temporary_path, path_to_graph)
            temporary_path = None
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)

    def _collect_interactions(self, train_sampler, validation_sampler, test_sampler):
        train_interactions = []
        train_user_interactions, train_item_interactions = [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        samplers_to_process = [train_sampler]
        if not self._use_train_data_only:
            samplers_to_process.extend([validation_sampler, test_sampler])

        for sampler in samplers_to_process:
            for sample in sampler.dataset:
                if 'item.ids' not in sample:
                    raise ValueError(
                        'unsupported graph sample: item.ids is required; raw evaluation history/target rows are not graph inputs'
                    )
                user_id = sample['user.ids'][0]
                for item_id in sample['item.ids']:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

        return {
            'train_interactions': train_interactions,
            'train_user_interactions': train_user_interactions,
            'train_item_interactions': train_item_interactions,
            'train_user_2_items': train_user_2_items,
            'train_item_2_users': train_item_2_users,
        }

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            graph_dir_path=config['graph_dir_path'],
            use_train_data_only=config.get('use_train_data_only', True),
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False),
            neighborhood_size=config.get('neighborhood_size', None),
            interaction_weighting=config.get('interaction_weighting', 'binary'),
        )

    @staticmethod
    def get_sparse_graph_layer(
        sparse_matrix,
        fst_dim,
        snd_dim,
        biparite=False,
    ):
        if not biparite:
            adj_mat = sparse_matrix.tocsr()
        else:
            R = sparse_matrix.tocsr()

            upper_right = R
            lower_left = R.T

            upper_left = sp.csr_matrix((fst_dim, fst_dim))
            lower_right = sp.csr_matrix((snd_dim, snd_dim))

            adj_mat = sp.bmat([[upper_left, upper_right], [lower_left, lower_right]])
            assert adj_mat.shape == (fst_dim + snd_dim, fst_dim + snd_dim), (
                f'Got shape {adj_mat.shape}, expected {(fst_dim + snd_dim, fst_dim + snd_dim)}'
            )

        rowsum = np.array(adj_mat.sum(1))
        with np.errstate(divide='ignore'):  # zero-degree rows (padding, users without edges) get 0 below
            d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    @staticmethod
    def _filter_matrix_by_top_k(matrix, k):
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError('top-k must be a positive integer')
        mat = matrix.tolil()

        for i in range(mat.shape[0]):
            if len(mat.rows[i]) <= k:
                continue
            data = np.array(mat.data[i])

            top_k_indices = np.argpartition(data, -k)[-k:]
            mat.data[i] = [mat.data[i][j] for j in top_k_indices]
            mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]

        return mat.tocsr()

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        meta = {
            'user_graph': self._user_graph,
            'item_graph': self._item_graph,
            'graph': self._graph,
            **self._dataset.meta,
        }
        return meta
