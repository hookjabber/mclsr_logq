"""Verify cached graph artifacts against the train ladder files: eq. 1 on the bipartite graph,
co-action weights + top-50 + eq. 1 on the user-user and item-item graphs. Standalone (scipy only).

Only the legacy 20-item train-row / 20-item eval-history protocol is supported.
Catalogue dimensions follow its all-split, post-crop maxima; targets are not cropped.
Manifest-backed v2 datasets are explicitly unsupported, not verified by this script.
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.sparse as sp


results = []


def check(name, diff, tolerance):
    passed = np.isfinite(diff) and diff <= tolerance
    results.append(passed)
    print(f'  {"OK  " if passed else "FAIL"} {name}: {diff:.3e} (tol {tolerance:g})')


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
    out.append(('kept edges that are not co-action pairs', Nb.nnz - K.nnz, 0))
    cnt_ref = np.minimum(np.diff(C.indptr), k)
    cnt_got = np.diff(N.indptr)
    out.append(('rows with kept count != min(50, #neighbours)', int((cnt_ref != cnt_got).sum()), 0))
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
    out.append(('rows whose kept weights != top-50 co-action weights', bad, 0))
    out.append(
        (
            'max |D^-1/2 K D^-1/2 - cached|',
            float(abs(sym_norm(K) - N).max()) if Nb.nnz == K.nnz else float('nan'),
            1e-6,
        )
    )
    return out


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('datasets', nargs='+', help='Dataset directory names under data/')
args = parser.parse_args()

for name in args.datasets:
    d = f'data/{name}'
    if os.path.exists(os.path.join(d, 'preprocessing.json')):
        parser.error('Unsupported verifier dataset: manifest-backed/v2 data; only legacy 20/20 is supported')
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
        n_items = max(n_items, max(items, default=0))
        for it in items:
            pairs.add((u, it))
    for split_file in ('valid_history.txt', 'valid_target.txt', 'test_history.txt', 'test_target.txt'):
        with open(os.path.join(d, split_file)) as source:
            for line in source:
                identifiers = [int(value) for value in line.split()]
                U = max(U, identifiers[0])
                items = identifiers[1:] if split_file.endswith('_target.txt') else identifiers[1:][-20:]
                n_items = max(n_items, max(items, default=0))
    G = sp.load_npz(f'{d}/general_graph_trainOnly.npz').tocsr()
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
    nu, nv = U + 2, n_items + 2
    print(
        f'  train pairs {len(pairs)}, all-split max user id {U}, max item id {n_items}; cached shapes: general {G.shape}, user {None if Nu is None else Nu.shape}, item {None if Nv is None else Nv.shape}'
    )
    shapes_match = True
    for graph_name, matrix, expected_shape in (
        ('general', G, (nu + nv, nu + nv)),
        ('user-user', Nu, (nu, nu)),
        ('item-item', Nv, (nv, nv)),
    ):
        if matrix is not None:
            matches = matrix.shape == expected_shape
            check(f'{graph_name}: shape matches all-split catalogue {expected_shape}', float(not matches), 0)
            shapes_match &= matches
    if not shapes_match:
        continue
    pu = np.array([p[0] for p in pairs])
    pi = np.array([p[1] for p in pairs])
    R = sp.csr_matrix((np.ones(len(pairs)), (pu, pi)), shape=(nu, nv))
    A = sp.bmat([[None, R], [R.T, None]]).tocsr()
    check('general: max |D^-1/2 A D^-1/2 - cached|', float(abs(sym_norm(A) - G).max()), 1e-6)
    check('general: sparsity pattern mismatch', ((A != 0) != (G != 0)).nnz, 0)
    if Nu is not None:
        for metric, diff, tolerance in check_sim('user', R @ R.T, Nu):
            check(f'user-user: {metric}', diff, tolerance)
    else:
        print('  SKIP user-user: no cached artifact')
    if Nv is not None:
        for metric, diff, tolerance in check_sim('item', R.T @ R, Nv):
            check(f'item-item: {metric}', diff, tolerance)
    else:
        print('  SKIP item-item: no cached artifact')

print(f'\n{sum(results)}/{len(results)} checks passed')
sys.exit(0 if results and all(results) else 1)
