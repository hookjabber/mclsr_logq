"""Verify cached graph artifacts against the train ladder files: eq. 1 on the bipartite graph,
co-action weights + top-50 + eq. 1 on the user-user and item-item graphs. Standalone (scipy only)."""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp


def sym_norm(A):
    A = A.tocsr().astype(np.float64)
    d = np.asarray(A.sum(1)).ravel()
    with np.errstate(divide='ignore'):
        dinv = np.where(d > 0, d**-0.5, 0.0)
    D = sp.diags(dinv)
    return (D @ A @ D).tocsr()


def check_sim(name, C, N, k=50):
    C = C.tocsr().astype(np.float64)
    C = (C - sp.diags(C.diagonal())).tocsr()
    C.eliminate_zeros()
    C.sort_indices()  # NB setdiag(0) corrupts CSR with empty rows in scipy 1.14
    N = N.tocsr()
    N.eliminate_zeros()
    Nb = (N != 0).astype(np.float64)
    K = C.multiply(Nb).tocsr()
    K.eliminate_zeros()
    out = []
    out.append(('kept edges that are not co-action pairs', Nb.nnz - K.nnz))
    cnt_ref = np.minimum(np.diff(C.indptr), k)
    cnt_got = np.diff(N.indptr)
    out.append(('rows with kept count != min(50, #neighbours)', int((cnt_ref != cnt_got).sum())))
    bad = 0
    for r in range(C.shape[0]):
        row = C.data[C.indptr[r] : C.indptr[r + 1]]
        if len(row) == 0:
            if N.indptr[r + 1] - N.indptr[r] > 0:
                bad += 1
            continue
        top = np.sort(row)[::-1][:k]
        kept = np.sort(K.data[K.indptr[r] : K.indptr[r + 1]])[::-1]
        if len(kept) != len(top) or np.abs(kept - top).max() > 0:
            bad += 1
    out.append(('rows whose kept weights != top-50 co-action weights', bad))
    out.append(
        ('max |D^-1/2 K D^-1/2 - cached|', float(abs(sym_norm(K) - N).max()) if Nb.nnz == K.nnz else float('nan'))
    )
    return out


for name in sys.argv[1:]:
    d = f'data/{name}'
    print(f'===== {name} =====')
    for f in sorted(os.listdir(d)):
        if f.endswith('.npz') or f == 'train_mclsr.txt':
            st = os.stat(os.path.join(d, f))
            print(f'  {f:40s} {st.st_size:>12d}  {time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}')
    pairs = set()
    U = n_items = 0
    for line in open(f'{d}/train_mclsr.txt'):
        p = line.split()
        u = int(p[0])
        items = [int(x) for x in p[1:]][-20:]
        U = max(U, u)
        for it in items:
            pairs.add((u, it))
            n_items = max(n_items, it)
    # the framework sizes rows by num_users/num_items from train AND eval files: read them from the cached shapes
    G = sp.load_npz(f'{d}/general_graph_trainOnly.npz').tocsr()
    n_total = G.shape[0]
    Nu = (
        sp.load_npz(f'{d}/user_graph_k50_trainOnly.npz').tocsr()
        if os.path.exists(f'{d}/user_graph_k50_trainOnly.npz')
        else None
    )
    Nv = (
        sp.load_npz(f'{d}/item_graph_k50_trainOnly.npz').tocsr()
        if os.path.exists(f'{d}/item_graph_k50_trainOnly.npz')
        else None
    )
    nu = Nu.shape[0] if Nu is not None else U + 2
    nv = n_total - nu
    print(
        f'  train pairs {len(pairs)}, max user id {U}, max item id {n_items}; cached shapes: general {G.shape}, user {None if Nu is None else Nu.shape}, item {None if Nv is None else Nv.shape}'
    )
    pu = np.array([p[0] for p in pairs])
    pi = np.array([p[1] for p in pairs])
    R = sp.csr_matrix((np.ones(len(pairs)), (pu, pi)), shape=(nu, nv))
    A = sp.bmat([[None, R], [R.T, None]]).tocsr()
    print(
        f'  general: max |D^-1/2 A D^-1/2 - cached| = {abs(sym_norm(A) - G).max():.2e}, nnz ref {A.nnz} vs cached {G.nnz}'
    )
    if Nu is not None:
        for k, v in check_sim('user', R @ R.T, Nu):
            print(f'  user-user: {k}: {v}')
    if Nv is not None:
        for k, v in check_sim('item', R.T @ R, Nv):
            print(f'  item-item: {k}: {v}')
