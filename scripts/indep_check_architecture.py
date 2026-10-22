"""Independent end-to-end check of the MCLSR architecture on a real Clothing batch.

Everything on the reference side is recomputed from the raw text files and the model's
state_dict with numpy / scipy in float64 -- no model code is reused. Compared against
the framework: three graphs (eq. 1, §3.2, K=50), eq. 2-7, 9-10, 13 (model forward),
eq. 16 (eval top-k), eq. 14 in-batch L_P (lambda 0 / 1), eq. 8 / 11 (paper scheme with
cosine) and the default symmetric form, and the composite objective eq. 15.

Scope: full MCLSR with train-only top-50 graphs, two last-layer propagation steps,
and the legacy 20-item train-row protocol. Configured alpha, shared/separate interest
projectors, optional LayerNorm and its epsilon are supported. The retrieval reference
assumes dot scores, unit temperature, same-user masking and negatives-only logQ;
this is not a universal verifier for other model, loss or preprocessing variants.
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time

sys.path.insert(0, 'src')
import numpy as np
import scipy.sparse as sp
import torch
from irec.utils import DEVICE, fix_random_seed  # noqa: E402
from irec.dataset import BaseDataset  # noqa: E402
from irec.dataloader import BaseDataloader  # noqa: E402
from irec.models import BaseModel  # noqa: E402
from irec.loss import BaseLoss  # noqa: E402

t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.1f}s]', *a, flush=True)


results = []


def check(name, diff, tol):
    ok = diff <= tol
    results.append((name, diff, tol, ok))
    log(f'{"OK " if ok else "FAIL"} {name}: max|diff| = {diff:.3e} (tol {tol:g})')


ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/train/clothing64/05_full_l00.json')
ap.add_argument('--data', default='data/Clothing')
args = ap.parse_args()
cfg = json.load(open(args.config))
DATA = args.data
if os.path.exists(os.path.join(DATA, 'preprocessing.json')):
    ap.error('Unsupported verifier dataset: manifest-backed/v2 data; only legacy 20/20 is supported')
if cfg['model'].get('num_graph_layers', 0) != 2 or cfg['model'].get('graph_layer_mean', False):
    ap.error('Unsupported verifier configuration: requires two graph layers and last-layer propagation')
fix_random_seed(1)
log('building dataset (user graph may be built from scratch)...')
dataset = BaseDataset.create_from_config(cfg['dataset'])
meta = dataset.meta
U, n_items = meta['num_users'], meta['num_items']
train_sampler, val_sampler, test_sampler = dataset.get_samplers()
log(f'dataset ready: users={U} items={n_items}')


# ---------- raw data ----------
def read_lines(path, max_len=None):
    out = []
    for line in open(path):
        p = line.split()
        items = [int(x) for x in p[1:]]
        if max_len:
            items = items[-max_len:]
        out.append((int(p[0]), items))
    return out


train_lines = read_lines(f'{DATA}/train_mclsr.txt', 20)
print('min/max items per ladder line:', min(len(s) for _, s in train_lines), max(len(s) for _, s in train_lines))
pairs = sorted({(u, it) for u, items in train_lines for it in items})
pu = np.array([p[0] for p in pairs])
pi = np.array([p[1] for p in pairs])
R = sp.csr_matrix((np.ones(len(pairs)), (pu, pi)), shape=(U + 2, n_items + 2))
print('unique (user,item) train pairs:', len(pairs))


def sym_norm(A):
    A = A.tocsr().astype(np.float64)
    d = np.asarray(A.sum(1)).ravel()
    with np.errstate(divide='ignore'):
        dinv = np.where(d > 0, d**-0.5, 0.0)
    D = sp.diags(dinv)
    return (D @ A @ D).tocsr()


def torch_sparse_to_scipy(t):
    t = t.coalesce().cpu()
    idx = t.indices().numpy()
    val = t.values().double().numpy()
    return sp.csr_matrix((val, (idx[0], idx[1])), shape=tuple(t.shape))


# eq. 1 on the bipartite graph
A_uv = sp.bmat([[None, R], [R.T, None]]).tocsr()
N_uv_ref = sym_norm(A_uv)
N_uv = torch_sparse_to_scipy(meta['graph'])
check('bipartite graph: D^-1/2 A D^-1/2 vs framework', abs(N_uv_ref - N_uv).max(), 1e-6)
check('bipartite graph: same sparsity pattern', float((N_uv_ref != 0).astype(int).multiply(1).nnz != N_uv.nnz), 0)


# user-user / item-item: weights = co-action counts, top-50 per row, eq. 1
def check_similarity_graph(name, C, N):
    C = C.tocsr().astype(np.float64)
    C = (C - sp.diags(C.diagonal())).tocsr()
    C.eliminate_zeros()
    C.sort_indices()  # NB setdiag(0) corrupts CSR with empty rows in scipy 1.14
    N = N.tocsr()
    N.eliminate_zeros()
    Nb = (N != 0).astype(np.float64)
    K = C.multiply(Nb).tocsr()
    K.eliminate_zeros()  # co-action weights on the kept pattern
    check(
        f'{name}: kept edges are co-action pairs', float(Nb.nnz - K.nnz), 0
    )  # a kept edge with C == 0 vanishes from K
    k = 50
    cnt_ref = np.minimum(np.diff(C.indptr), k)
    cnt_got = np.diff(N.indptr)
    check(f'{name}: per-row kept count = min(50, #neighbours)', float(np.abs(cnt_ref - cnt_got).max()), 0)
    worst = 0.0
    for r in range(C.shape[0]):
        row = C.data[C.indptr[r] : C.indptr[r + 1]]
        if len(row) == 0:
            continue
        top = np.sort(row)[::-1][:k]
        kept = np.sort(K.data[K.indptr[r] : K.indptr[r + 1]])[::-1]
        if len(kept) != len(top):
            worst = np.inf
            break
        worst = max(worst, float(np.abs(kept - top).max()))
    check(f'{name}: kept weights = top-50 co-action weights per row (ties ignored)', worst, 0)
    check(f'{name}: normalisation = D^-1/2 K D^-1/2 of the kept weights', abs(sym_norm(K) - N).max(), 1e-6)
    return K


C_uu = (R @ R.T).tocsr()
C_vv = (R.T @ R).tocsr()
N_uu = torch_sparse_to_scipy(meta['user_graph'])
N_vv = torch_sparse_to_scipy(meta['item_graph'])
check_similarity_graph('user-user', C_uu, N_uu)
check_similarity_graph('item-item', C_vv, N_vv)
print(
    'user-user: rows with >50 neighbours:',
    int((np.diff(C_uu.indptr) > 50).sum()),
    'of',
    U,
    '| item-item: rows with >50 neighbours:',
    int((np.diff(C_vv.indptr) > 50).sum()),
    'of',
    n_items,
)

# ---------- model forward on a real batch (dropout and edge dropout off) ----------
mcfg = copy.deepcopy(cfg['model'])
mcfg['dropout'] = 0.0
mcfg['graph_dropout'] = 0.0
model = BaseModel.create_from_config(mcfg, **meta).to(DEVICE)
dl = BaseDataloader.create_from_config(cfg['dataloader']['train'], dataset=train_sampler, **meta)
batch = next(iter(dl))
batch = {k: v.to(DEVICE) for k, v in batch.items()}
model.train()
with torch.no_grad():
    out = model(dict(batch))
sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
E_i = sd['_item_embeddings.weight']
E_u = sd['_user_embeddings.weight']
P = sd['_position_embeddings.weight']
if mcfg.get('use_layernorm', True):
    ln_w, ln_b = sd['_layernorm.weight'], sd['_layernorm.bias']
    ln_eps = mcfg.get('layer_norm_eps', 1e-5)
W1 = sd['_current_interest_learning_encoder.0.weight']
W2 = sd['_current_interest_learning_encoder.2.weight']
W3 = sd['_general_interest_learning_encoder.0.weight']


def proj(prefix, x):
    a, b = sd[f'{prefix}.0.weight'], sd[f'{prefix}.0.bias']
    c, d = sd[f'{prefix}.2.weight'], sd[f'{prefix}.2.bias']
    h = x @ a.T + b
    h = np.where(h > 0, h, np.exp(np.minimum(h, 0)) - 1)
    return h @ c.T + d


def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


ids = batch['item.ids'].cpu().numpy()
lens = batch['item.length'].cpu().numpy()
users = batch['user.ids'].cpu().numpy()
labels = batch['labels.ids'].cpu().numpy()
B = len(lens)
offs = np.concatenate([[0], np.cumsum(lens)])
seqs = [ids[offs[b] : offs[b + 1]] for b in range(B)]
print(
    f'batch: B={B}, lengths {lens.min()}..{lens.max()}, distinct users {len(set(users))}, distinct labels {len(set(labels))}'
)

# eq. 4 / 9: propagation in float64 scipy
X = np.vstack([E_u, E_i])
H = N_uv_ref @ (N_uv_ref @ X)
H_u, H_i = H[: U + 2], H[U + 2 :]
Huu = N_uu @ (N_uu @ E_u)
Hvv = N_vv @ (N_vv @ E_i)

I_s = np.zeros((B, E_i.shape[1]))
I_c = np.zeros_like(I_s)
for b, seq in enumerate(seqs):
    L = len(seq)
    e = E_i[seq]
    x = e + P[np.arange(L - 1, -1, -1)]  # E_{u,p}: last item gets position 0
    if mcfg.get('use_layernorm', True):
        mu = x.mean(1, keepdims=True)
        var = x.var(1, keepdims=True)
        x = (x - mu) / np.sqrt(var + ln_eps) * ln_w + ln_b
    a_s = softmax((W2 @ np.tanh(W1 @ x.T)).ravel())  # eq. 2
    I_s[b] = a_s @ e  # eq. 3 (raw embeddings)
    Es = H_i[seq]
    h_u = H_u[users[b]]
    a_c = softmax(Es @ np.tanh(W3 @ h_u))  # eq. 5
    I_c[b] = a_c @ Es  # eq. 6
alpha = mcfg.get('alpha', 0.5)
I_comb = alpha * I_s + (1 - alpha) * I_c


def g(k):
    return out[k].detach().cpu().double().numpy()


check('eq. 2-6, 13: combined representation', np.abs(g('combined_representation') - I_comb).max(), 1e-5)
check(
    'eq. 7: projected sequential view',
    np.abs(g('sequential_representation') - proj('_sequential_projector', I_s)).max(),
    1e-5,
)
graph_projector = '_sequential_projector' if mcfg.get('share_il_projector', False) else '_graph_projector'
check('eq. 7: projected graph view', np.abs(g('graph_representation') - proj(graph_projector, I_c)).max(), 1e-5)
check(
    'label representation = raw item embedding of the target',
    np.abs(g('label_representation') - E_i[labels]).max(),
    1e-7,
)
check(
    'eq. 9-10: user-user view (projected)',
    np.abs(g('user_graph_user_embeddings') - proj('_user_projection', Huu[users])).max(),
    1e-5,
)
check(
    'eq. 9-10: user-item view of users (projected)',
    np.abs(g('common_graph_user_embeddings') - proj('_user_projection', H_u[users])).max(),
    1e-5,
)
uniq = np.unique(ids)
check('L_IC item ids = unique batch items', float(np.abs(g('item_ids') - uniq).max()), 0)
check(
    'eq. 9-10: item-item view (projected)',
    np.abs(g('item_graph_item_embeddings') - proj('_item_projection', Hvv[uniq])).max(),
    1e-5,
)
check(
    'eq. 9-10: user-item view of items (projected)',
    np.abs(g('common_graph_item_embeddings') - proj('_item_projection', H_i[uniq])).max(),
    1e-5,
)

# eq. 16: eval top-k from I_s only, full catalogue
model.eval()
with torch.no_grad():
    idx = model(dict(batch)).cpu().numpy()
scores = I_s @ E_i.T
scores[:, 0] = -np.inf
scores[:, n_items + 1 :] = -np.inf
worst = 0.0
for b in range(B):
    top = np.sort(scores[b])[::-1][: idx.shape[1]]
    worst = max(worst, float(np.abs(np.sort(scores[b, idx[b]])[::-1] - top).max()))
check(f'eq. 16: eval top-{idx.shape[1]} scores from I_s (ties ignored)', worst, 1e-5)


# ---------- losses on the real outputs ----------
def ce_rows(S, mask_invalid):  # mean over rows of -log softmax at the diagonal
    S = S.copy()
    S[mask_invalid] = -1e12
    S = S - S.max(1, keepdims=True)
    return float(np.mean(-np.diag(S) + np.log(np.exp(S).sum(1))))


same_item = labels[None, :] == labels[:, None]
same_user = users[None, :] == users[:, None]
invalid = (same_item | same_user) & ~np.eye(B, dtype=bool)
S = I_comb @ E_i[labels].T
lp_cfg = copy.deepcopy(cfg['loss']['losses'][0])
lp0 = float(BaseLoss.create_from_config(lp_cfg)(dict(out)))
check('eq. 14 in-batch L_P, lambda=0', abs(lp0 - ce_rows(S, invalid)), 1e-5)
counts = pickle.load(open(cfg['loss']['losses'][0]['path_to_item_counts'], 'rb'))
counts = np.asarray(counts['counts'] if isinstance(counts, dict) else counts, dtype=np.float64)
logq = np.log(np.clip(counts / counts.sum(), 1e-10, None))
S1 = S - logq[labels][None, :]
np.fill_diagonal(S1, np.diag(S))  # negatives_only: positive uncorrected
lp_cfg['logq_lambda'] = 1.0
lp1 = float(BaseLoss.create_from_config(lp_cfg)(dict(out)))
check('eq. 14 in-batch L_P + logQ (negatives only), lambda=1', abs(lp1 - ce_rows(S1, invalid)), 1e-5)


def paper_scheme(f, s, tau, cosine):  # eq. 8 / 11: anchor = f_i, candidates = all s_j + f_{j != i}
    if cosine:
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        s = s / np.linalg.norm(s, axis=1, keepdims=True)
    S = np.concatenate([f @ s.T, f @ f.T], 1) / tau
    inv = np.zeros_like(S, dtype=bool)
    inv[np.arange(len(f)), len(f) + np.arange(len(f))] = True
    return ce_rows(S, inv)


def symmetric_scheme(f, s, tau):  # framework default: NT-Xent over 2B anchors, / 2
    Z = np.concatenate([f, s])
    n = len(f)
    S = Z @ Z.T / tau
    inv = np.eye(2 * n, dtype=bool)
    S[inv] = -1e12
    S = S - S.max(1, keepdims=True)
    pos = np.concatenate([np.diag(S[:n, n:]), np.diag(S[n:, :n])])
    return float(np.mean(-pos + np.log(np.exp(S).sum(1))) / 2)


views = {
    'L_IL': ('sequential_representation', 'graph_representation'),
    'L_UC (anchor uv)': ('common_graph_user_embeddings', 'user_graph_user_embeddings'),
    'L_IC (anchor uv)': ('common_graph_item_embeddings', 'item_graph_item_embeddings'),
}
for name, (fk, sk) in views.items():
    f, s = g(fk), g(sk)
    got_sym = float(
        BaseLoss.create_from_config(
            {'type': 'fps', 'fst_embeddings_prefix': fk, 'snd_embeddings_prefix': sk, 'temperature': 0.5}
        )(dict(out))
    )
    check(f'{name}: default symmetric dot form', abs(got_sym - symmetric_scheme(f, s, 0.5)), 1e-4)
    got_paper = float(
        BaseLoss.create_from_config(
            {
                'type': 'fps',
                'fst_embeddings_prefix': fk,
                'snd_embeddings_prefix': sk,
                'temperature': 0.5,
                'scheme': 'paper',
                'normalize_embeddings': True,
            }
        )(dict(out))
    )
    check(f'{name}: paper form (eq. 8/11, cosine, tau 0.5)', abs(got_paper - paper_scheme(f, s, 0.5, True)), 1e-5)
    print(f'      {name}: symmetric {got_sym:.4f} vs paper {got_paper:.4f}')

# eq. 15 composite
total = float(BaseLoss.create_from_config(cfg['loss'])(dict(out)))
parts = [float(BaseLoss.create_from_config(line)(dict(out))) for line in cfg['loss']['losses']]
w = [line.get('weight', 1.0) for line in cfg['loss']['losses']]
check(
    'eq. 15: composite = L_P + 1 L_IL + 0.05 L_UC + 0.05 L_IC',
    abs(total - sum(wi * p for wi, p in zip(w, parts))),
    1e-5,
)

# duplicate users in this batch (L_UC false negatives)
print('rows sharing a user with another row in this batch:', int((same_user & ~np.eye(B, dtype=bool)).any(1).sum()))

# eval split sanity: target share of the full sequence for test users
th = read_lines(f'{DATA}/test_history.txt')
tt = dict(read_lines(f'{DATA}/test_target.txt'))
shares = [len(tt[u]) / (len(h) + len(tt[u])) for u, h in th if u in tt]
print(
    f'test users: {len(shares)}, target share of sequence: mean {np.mean(shares):.3f}, min {min(shares):.3f}, max {max(shares):.3f}, history len max {max(len(h) for _, h in th)}'
)
tr_users = {u for u, _ in train_lines}
va = {u for u, _ in read_lines(f'{DATA}/valid_history.txt')}
te = {u for u, _ in th}
print('user overlap train/valid/test:', len(tr_users & va), len(tr_users & te), len(va & te))

print('\n==== SUMMARY ====')
for name, diff, tol, ok in results:
    print(f'{"OK  " if ok else "FAIL"} {name}  ({diff:.2e})')
print(f'{sum(ok for *_, ok in results)}/{len(results)} checks passed')
sys.exit(0 if results and all(ok for *_, ok in results) else 1)
