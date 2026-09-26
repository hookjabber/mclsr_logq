"""Why the feature-level graphs contribute nothing: (1) the user–user / item–item graphs are the
top-k truncation of the 2-hop projection of the user–item graph that the LightGCN branch already
propagates over; (2) at a trained checkpoint the two "views" contrasted by L_UC / L_IC are nearly
identical, while the L_IL pair (sequence vs graph) is not; (3) the feature-level losses barely move
during training.  CPU only.

    CUDA_VISIBLE_DEVICES= python scripts/diagnose_feature_graphs.py \
        --params configs/train/beauty/05_full_baseline.json \
        --checkpoint checkpoints/beauty_mclsr_grid_05_full_baseline_Beauty_confirm_seed1_best_validation_ndcg_at_20.pth
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from irec.dataloader import BaseDataloader  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.utils import DEVICE, fix_random_seed  # noqa: E402


def graph_overlap(data_dir, k=50):
    general = sp.load_npz(os.path.join(data_dir, 'general_graph_trainOnly.npz')).tocsr()
    uu = sp.load_npz(os.path.join(data_dir, f'user_graph_k{k}_trainOnly.npz')).tocsr()
    ii = sp.load_npz(os.path.join(data_dir, f'item_graph_k{k}_trainOnly.npz')).tocsr()
    num_users = uu.shape[0]  # the graphs carry the padding / mask rows, so take the size from the graph itself
    A = general[:num_users, num_users:].tocsr()  # users x items
    out = {}
    for name, stored, proj in (('user-user', uu, A @ A.T), ('item-item', ii, A.T @ A)):
        proj = proj.tolil()
        proj.setdiag(0)
        proj = proj.tocsr()
        covered, total, jacc, n = 0, 0, 0.0, 0
        for r in range(stored.shape[0]):
            s = set(stored.indices[stored.indptr[r] : stored.indptr[r + 1]])
            if not s:
                continue
            row = proj.getrow(r)
            two_hop = set(row.indices)
            covered += len(s & two_hop)
            total += len(s)
            if row.nnz:
                order = np.argsort(-row.data)[:k]
                topk = set(row.indices[order])
                jacc += len(s & topk) / len(s | topk)
                n += 1
        out[name] = {
            'stored_neighbours_inside_2hop': covered / max(total, 1),
            'mean_jaccard_with_top%d_of_2hop' % k: jacc / max(n, 1),
            'rows': stored.shape[0],
        }
    return out


def view_similarity(config, checkpoint, batches=8):
    fix_random_seed(1)
    dataset = BaseDataset.create_from_config(config['dataset'])
    train_sampler, _, _ = dataset.get_samplers()
    loader = BaseDataloader.create_from_config(config['dataloader']['train'], dataset=train_sampler, **dataset.meta)
    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state.get('model_state_dict', state))
    model.train()  # the training branch emits the contrastive views
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    if hasattr(model, '_graph_dropout'):
        model._graph_dropout = 0.0
    pairs = {
        'L_UC pair: user–user graph view vs user–item graph view (same user)': (
            'user_graph_user_embeddings',
            'common_graph_user_embeddings',
        ),
        'L_IC pair: item–item graph view vs user–item graph view (same item)': (
            'item_graph_item_embeddings',
            'common_graph_item_embeddings',
        ),
        'L_IL pair: sequential interest vs graph interest (same user)': (
            'sequential_representation',
            'graph_representation',
        ),
    }
    acc = {k: {'same': [], 'other': []} for k in pairs}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch)
            for name, (a, b) in pairs.items():
                x = torch.nn.functional.normalize(out[a], dim=-1)
                y = torch.nn.functional.normalize(out[b], dim=-1)
                sim = x @ y.T
                acc[name]['same'].append(sim.diagonal().mean().item())
                off = sim - torch.diag(sim.diagonal())
                acc[name]['other'].append((off.sum() / (sim.numel() - sim.shape[0])).item())
    return {
        k: {'cos_same_entity': float(np.mean(v['same'])), 'cos_other_entities': float(np.mean(v['other']))}
        for k, v in acc.items()
    }


def loss_curves(experiment_name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs = sorted(
        glob.glob(f'tensorboard_logs/{experiment_name}_confirm_seed1_*')
        or glob.glob(f'tensorboard_logs/{experiment_name}_*')
    )
    if not runs:
        return {}
    acc = EventAccumulator(runs[-1], size_guidance={'scalars': 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()['scalars']:
        if 'loss' not in tag or 'validation' in tag or 'eval' in tag:
            continue
        v = np.array([e.value for e in acc.Scalars(tag)])
        if len(v) < 200:
            continue
        out[tag] = {
            'first_100_steps': float(v[:100].mean()),
            'last_500_steps': float(v[-500:].mean()),
            'min': float(v.min()),
            'steps': int(len(v)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    config = json.load(open(args.params))
    data_dir = (
        os.path.join(config['dataset']['path_to_data_dir'], config['dataset']['name'])
        if 'name' in config['dataset']
        else config['dataset'].get('graph_dir_path')
    )
    report = {
        'graph_overlap': graph_overlap(data_dir, config['dataset'].get('neighborhood_size', 50)),
        'view_similarity_at_checkpoint': view_similarity(config, args.checkpoint),
        'training_loss_curves': loss_curves(config['experiment_name']),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
