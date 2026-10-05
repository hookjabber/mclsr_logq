"""Independent check of the SASRec branch on a real batch.

The transformer encoder is re-implemented in numpy / float64 from the model's ``state_dict``
(post-LN ``nn.TransformerEncoderLayer``: causal multi-head self-attention with key padding, ReLU
feed-forward, residuals), and the following are compared with the framework:

* the evaluation inputs of the SASRec dataset are identical to those of the MCLSR dataset for the
  same data directory (same users in the same order, same histories, same targets);
* the per-position encoder outputs and the eval top-k from the last valid position (eq. 16 protocol);
* causality (changing the item at position t changes no output before t) and padding invariance
  (a sequence scored alone equals the same sequence inside a batch of longer sequences);
* the training outputs (one query per position, positive = next item, owner user id per query) and
  the in-batch retrieval loss on them against a hand-written reference.

    PYTHONPATH=src python scripts/indep_check_sasrec.py --config configs/train/clothing64/sasrec_inbatch_l00.json \
        --mclsr-config configs/train/clothing64/01_orig.json
"""

import argparse
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')
from irec.dataloader import BaseDataloader  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.loss import BaseLoss  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.utils import DEVICE, fix_random_seed  # noqa: E402

results = []


def check(name, diff, tol):
    ok = diff <= tol
    results.append((name, diff, tol, ok))
    print(f'{"OK " if ok else "FAIL"} {name}: max|diff| = {diff:.3e} (tol {tol:g})', flush=True)


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def encoder_numpy(sd, items, num_heads, eps):
    """Outputs for ONE sequence (no padding): (L, d)."""
    E, P = sd['_item_embeddings.weight'], sd['_position_embeddings.weight']
    L = len(items)
    x = E[items] + P[np.arange(L - 1, -1, -1)]
    x = layer_norm(x, sd['_layernorm.weight'], sd['_layernorm.bias'], eps)
    d = x.shape[1]
    hd = d // num_heads
    causal = np.tril(np.ones((L, L), dtype=bool))
    layer = 0
    while f'_encoder.layers.{layer}.self_attn.in_proj_weight' in sd:
        p = f'_encoder.layers.{layer}.'
        qkv = x @ sd[p + 'self_attn.in_proj_weight'].T + sd[p + 'self_attn.in_proj_bias']
        q, k, v = np.split(qkv, 3, axis=1)
        heads = []
        for h in range(num_heads):
            sl = slice(h * hd, (h + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            s = np.where(causal, s, -np.inf)
            s = s - s.max(1, keepdims=True)
            a = np.exp(s)
            a = a / a.sum(1, keepdims=True)
            heads.append(a @ v[:, sl])
        attn = np.concatenate(heads, axis=1) @ sd[p + 'self_attn.out_proj.weight'].T + sd[p + 'self_attn.out_proj.bias']
        x = layer_norm(x + attn, sd[p + 'norm1.weight'], sd[p + 'norm1.bias'], eps)
        ff = np.maximum(x @ sd[p + 'linear1.weight'].T + sd[p + 'linear1.bias'], 0.0)
        ff = ff @ sd[p + 'linear2.weight'].T + sd[p + 'linear2.bias']
        x = layer_norm(x + ff, sd[p + 'norm2.weight'], sd[p + 'norm2.bias'], eps)
        layer += 1
    return x


def ce_rows(S, invalid):
    S = S.copy()
    S[invalid] = -1e12
    S = S - S.max(1, keepdims=True)
    return float(np.mean(-np.diag(S) + np.log(np.exp(S).sum(1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/train/clothing64/sasrec_inbatch_l00.json')
    ap.add_argument('--mclsr-config', default='configs/train/clothing64/01_orig.json')
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    fix_random_seed(1)
    dataset = BaseDataset.create_from_config(cfg['dataset'])
    meta = dataset.meta
    train_sampler, _, test_sampler = dataset.get_samplers()

    # 1. identical evaluation inputs for both encoders
    mcfg = json.load(open(args.mclsr_config))
    _, _, mclsr_test = BaseDataset.create_from_config(mcfg['dataset']).get_samplers()
    same = len(test_sampler) == len(mclsr_test) and all(
        a['user.ids'] == b['user.ids'] and a['item.ids'] == b['item.ids'] and a['labels.ids'] == b['labels.ids']
        for a, b in zip(test_sampler, mclsr_test)
    )
    check('eval inputs identical to the MCLSR dataset (users, histories, targets)', 0.0 if same else 1.0, 0)
    print(f'test users: {len(test_sampler)}')

    model_cfg = copy.deepcopy(cfg['model'])
    model_cfg['dropout'] = 0.0
    model = BaseModel.create_from_config(model_cfg, **meta).to(DEVICE)
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    eps = model_cfg.get('layer_norm_eps', 1e-9)
    num_heads = model_cfg['num_heads']

    # 2. encoder outputs on a real training batch (eval mode: no dropout), per sequence
    loader = BaseDataloader.create_from_config(cfg['dataloader']['train'], dataset=train_sampler, **meta)
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    ids, lens = batch['item.ids'].cpu().numpy(), batch['item.length'].cpu().numpy()
    offs = np.concatenate([[0], np.cumsum(lens)])
    seqs = [ids[offs[i] : offs[i + 1]] for i in range(len(lens))]
    model.eval()
    with torch.no_grad():
        emb, mask = model._apply_sequential_encoder(batch['item.ids'], batch['item.length'])
    emb = emb.cpu().double().numpy()
    worst = 0.0
    for i, seq in enumerate(seqs):
        worst = max(worst, np.abs(emb[i, : len(seq)] - encoder_numpy(sd, seq, num_heads, eps)).max())
    print(f'batch: {len(seqs)} sequences, lengths {lens.min()}..{lens.max()}')
    check('numpy transformer == framework encoder at every valid position', worst, 1e-5)

    # 3. eval top-k from the last valid position (eq. 16 protocol)
    with torch.no_grad():
        idx = model(dict(batch)).cpu().numpy()
    E = sd['_item_embeddings.weight']
    n_items = meta['num_items']
    worst = 0.0
    for i, seq in enumerate(seqs):
        scores = encoder_numpy(sd, seq, num_heads, eps)[-1] @ E.T
        scores[0] = -np.inf
        scores[n_items + 1 :] = -np.inf
        top = np.sort(scores)[::-1][: idx.shape[1]]
        worst = max(worst, float(np.abs(np.sort(scores[idx[i]])[::-1] - top).max()))
    check(f'eval top-{idx.shape[1]} from the last position (ties ignored)', worst, 1e-5)

    # 4. causality: perturb position t, outputs before t must not move
    i = int(np.argmax(lens))
    seq = seqs[i].copy()
    t = len(seq) // 2
    seq2 = seq.copy()
    seq2[t] = seq[t] % n_items + 1  # some other item
    out2 = encoder_numpy(sd, seq2, num_heads, eps)
    with torch.no_grad():
        e1, m1 = model._apply_sequential_encoder(
            torch.tensor(seq, device=DEVICE), torch.tensor([len(seq)], device=DEVICE)
        )
        e2, m2 = model._apply_sequential_encoder(
            torch.tensor(seq2, device=DEVICE), torch.tensor([len(seq2)], device=DEVICE)
        )
    e1, e2 = e1[0].cpu().double().numpy(), e2[0].cpu().double().numpy()
    check(
        'causality in the framework: positions before a perturbed item are unchanged',
        np.abs(e1[:t] - e2[:t]).max(),
        1e-7,
    )
    check(
        'causality: the perturbed and later positions do change',
        0.0 if np.abs(e1[t:] - e2[t:]).max() > 1e-4 else 1.0,
        0,
    )
    check('numpy reference agrees on the perturbed sequence', np.abs(out2 - e2).max(), 1e-5)

    # 5. padding invariance: shortest sequence alone vs inside the batch
    j = int(np.argmin(lens))
    with torch.no_grad():
        alone, _ = model._apply_sequential_encoder(
            torch.tensor(seqs[j], device=DEVICE), torch.tensor([len(seqs[j])], device=DEVICE)
        )
    check(
        'padding invariance: a sequence alone == the same sequence inside a padded batch',
        np.abs(alone[0].cpu().double().numpy() - emb[j, : len(seqs[j])]).max(),
        1e-6,
    )

    # 6. training outputs: one query per position, next item as positive, owner id per query
    model.train()
    with torch.no_grad():
        out = model(dict(batch))
    q = out['query_embeddings'].cpu().double().numpy()
    pos_ids = out['positive_ids'].cpu().numpy()
    uid = out['user_ids'].cpu().numpy()
    check(
        'training queries = encoder outputs at every position (dropout off)',
        np.abs(q - emb[mask.cpu().numpy()]).max(),
        1e-6,
    )
    check(
        'training positives = the next item of every position',
        float(np.abs(pos_ids - batch['positive.ids'].cpu().numpy()).max()),
        0,
    )
    check(
        'owner user id repeated per position',
        float(np.abs(uid - np.repeat(batch['user.ids'].cpu().numpy(), lens)).max()),
        0,
    )
    print(f'queries per batch: {len(q)} (MCLSR ladder would give {len(lens)})')

    # 7. in-batch retrieval loss on these outputs, lambda = 0 and 1, against a hand-written reference
    lcfg = copy.deepcopy(cfg['loss']['losses'][0])
    S = q @ E[pos_ids].T
    invalid = (pos_ids[None, :] == pos_ids[:, None]) & ~np.eye(len(pos_ids), dtype=bool)
    if lcfg.get('user_ids_prefix'):
        invalid |= (uid[None, :] == uid[:, None]) & ~np.eye(len(pos_ids), dtype=bool)
    lcfg['logq_lambda'] = 0.0
    check(
        'eq. 14 in-batch loss on SASRec queries, lambda = 0',
        abs(float(BaseLoss.create_from_config(lcfg)(dict(out))) - ce_rows(S, invalid)),
        1e-5,
    )
    import pickle

    counts = pickle.load(open(lcfg['path_to_item_counts'], 'rb'))
    counts = np.asarray(counts['counts'] if isinstance(counts, dict) else counts, dtype=np.float64)
    logq = np.log(np.clip(counts / counts.sum(), 1e-10, None))
    S1 = S - logq[pos_ids][None, :]
    np.fill_diagonal(S1, np.diag(S))
    lcfg['logq_lambda'] = 1.0
    check(
        'eq. 14 in-batch loss + logQ (negatives only), lambda = 1',
        abs(float(BaseLoss.create_from_config(lcfg)(dict(out))) - ce_rows(S1, invalid)),
        1e-5,
    )

    print('\n==== SUMMARY ====')
    for name, diff, tol, ok in results:
        print(f'{"OK  " if ok else "FAIL"} {name}  ({diff:.2e})')
    print(f'{sum(ok for *_, ok in results)}/{len(results)} checks passed')


if __name__ == '__main__':
    main()
