"""Data-level sanity checks for one dataset directory: the user split is disjoint, the logQ count
tables are reconstructed exactly from the train files (no valid/test interactions inside) and the
bipartite graph holds exactly the train (user, item) pairs.

    python scripts/check_split_leakage.py --data-dir data/Beauty
"""
import argparse, pickle, numpy as np, collections, scipy.sparse as sp
parser = argparse.ArgumentParser(); parser.add_argument('--data-dir', default='data/Beauty')
D = parser.parse_args().data_dir.rstrip('/') + '/'
def read(fn):
    rows = []
    for line in open(D + fn):
        t = line.split()
        if t: rows.append([int(x) for x in t])
    return rows
tr_s = read('train_sasrec.txt'); tr_m = read('train_mclsr.txt')
vh, vt, th, tt = read('valid_history.txt'), read('valid_target.txt'), read('test_history.txt'), read('test_target.txt')
print('lines: train_sasrec %d, train_mclsr(ladder) %d, valid %d/%d, test %d/%d' % (len(tr_s), len(tr_m), len(vh), len(vt), len(th), len(tt)))
print('sample train_sasrec:', tr_s[0][:8], '| ladder:', tr_m[0], tr_m[1], '| test_history:', th[0][:6], '| test_target:', tt[0][:6])
users_tr = {r[0] for r in tr_s}; users_v = {r[0] for r in vh}; users_t = {r[0] for r in th}
print('users: train %d, valid %d, test %d | train&valid %d, train&test %d, valid&test %d' % (
    len(users_tr), len(users_v), len(users_t), len(users_tr & users_v), len(users_tr & users_t), len(users_v & users_t)))
print('ladder users == train users:', {r[0] for r in tr_m} == users_tr, '| valid hist/target users equal:', {r[0] for r in vh} == {r[0] for r in vt}, '| test:', {r[0] for r in th} == {r[0] for r in tt})
# q-tables
ic = pickle.load(open(D + 'item_counts.pkl', 'rb')); tc = pickle.load(open(D + 'item_target_counts.pkl', 'rb')); cc = pickle.load(open(D + 'item_context_counts.pkl', 'rb'))
def unwrap(name, d):
    if isinstance(d, dict):
        meta = {k: v for k, v in d.items() if k != 'counts'}
        print('%s: dict with keys %s, meta %s' % (name, list(d.keys()), meta))
        return np.asarray(d['counts'], dtype=float)
    print('%s: plain array len %d (no role metadata)' % (name, len(d)))
    return np.asarray(d, dtype=float)
ic = unwrap('item_counts', ic); tc = unwrap('item_target_counts', tc); cc = unwrap('item_context_counts', cc)
all_items_train = collections.Counter(i for r in tr_s for i in r[1:])
targets_ladder = collections.Counter(r[-1] for r in tr_m)
context_ladder = collections.Counter(i for r in tr_m for i in r[1:-1])
def cmp(name, table, counter):
    v = np.zeros_like(table); 
    for k, c in counter.items(): v[k] += c
    nz = table.sum(); print('%s: table sum %.0f, recomputed sum %.0f, max|diff| %.0f, argmax-diff item %d (table %.0f vs recomputed %.0f)' % (
        name, nz, v.sum(), np.abs(table - v).max(), np.abs(table - v).argmax(), table[np.abs(table - v).argmax()], v[np.abs(table - v).argmax()]))
cmp('item_counts vs train_sasrec all positions', ic, all_items_train)
cmp('item_counts vs ladder targets', ic, targets_ladder)
cmp('item_target_counts vs ladder targets', tc, targets_ladder)
cmp('item_context_counts vs ladder contexts', cc, context_ladder)
# would val/test interactions change the table? (they must NOT be inside)
vt_items = collections.Counter(i for r in (vh + vt + th + tt) for i in r[1:])
print('items only in valid/test (count>0 there, 0 in train):', sum(1 for k in vt_items if k < len(ic) and all_items_train.get(k, 0) == 0),
      '| of them with nonzero item_counts:', sum(1 for k in vt_items if k < len(ic) and all_items_train.get(k, 0) == 0 and ic[k] > 0))
print('item id range train: %d..%d, table len %d' % (min(all_items_train), max(all_items_train), len(ic)))
# graph
g = sp.load_npz(D + 'general_graph_trainOnly.npz'); print('general_graph:', g.shape, 'nnz', g.nnz)
pairs = {(r[0], i) for r in tr_s for i in r[1:]}
print('distinct train (user,item) pairs:', len(pairs), '| nnz/2 =', g.nnz / 2, '| if graph is user-item bipartite symmetric, these match')
for fn in ('item_graph_k50_trainOnly.npz', 'user_graph_k50_trainOnly.npz'):
    m = sp.load_npz(D + fn); print(fn, m.shape, 'nnz', m.nnz)
