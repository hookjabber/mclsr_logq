"""Standalone replication of the logQ effect on Beauty without any irec code.
Own data loading (data/Beauty/*.txt), own attention-pooling encoder, own in-batch softmax with the
negatives-only logQ correction, own ndcg@20 / recall@1000. Run from the repo root:
    python scripts/indep_check_beauty.py 0.0 8 1   # lambda, epochs, seed
    python scripts/indep_check_beauty.py 1.0 12 1
Result 15.09.2026 (seed 1, CPU): lambda=0 -> test 0.0411 / 0.4080; lambda=1 -> 0.0619 / 0.5244 (+51 % / +29 %),
the same effect as the irec pipeline (0.0386 -> 0.0556, 0.436 -> 0.510 over three seeds)."""
import sys, math, random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
LAM = float(sys.argv[1]); EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
D = 'data/Beauty/'; L = 20; DIM = 64; B = 128
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.set_num_threads(8)
def read(fn): return [[int(x) for x in ln.split()] for ln in open(D + fn) if ln.strip()]
ladder = read('train_mclsr.txt'); seqs = read('train_sasrec.txt')
vh, vt, th, tt = read('valid_history.txt'), read('valid_target.txt'), read('test_history.txt'), read('test_target.txt')
N = max(max(r[1:]) for r in seqs); PAD, MASK = 0, N + 1
cnt = collections.Counter(i for r in seqs for i in r[1:]); counts = np.ones(N + 2); 
for k, v in cnt.items(): counts[k] = v
log_q = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32))  # (N+2,)
users = torch.tensor([r[0] for r in ladder]); targets = torch.tensor([r[-1] for r in ladder])
prefix = [r[1:-1][-L:] for r in ladder]
def pad(rows):  # right-pad to L, keep last L
    x = torch.zeros(len(rows), L, dtype=torch.long)
    for i, r in enumerate(rows):
        r = r[-L:]; x[i, :len(r)] = torch.tensor(r)
    return x
X = pad(prefix); print(f'items {N}, ladder {len(ladder)}, valid {len(vh)}, test {len(th)}, lambda {LAM}, seed {SEED}')
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(N + 2, DIM, padding_idx=0); s.pos = nn.Embedding(L, DIM)
        s.ln = nn.LayerNorm(DIM, eps=1e-9); s.drop = nn.Dropout(0.3); s.W1 = nn.Linear(DIM, DIM); s.w2 = nn.Linear(DIM, 1, bias=False)
        for p in (s.emb.weight, s.pos.weight, s.W1.weight, s.w2.weight): nn.init.normal_(p, std=0.02)
    def query(s, x):  # x (b, L) right-padded
        m = x > 0; h = s.emb(x) + s.pos(torch.arange(L)[None, :]); h = s.drop(s.ln(h))
        a = s.w2(torch.tanh(s.W1(h))).squeeze(-1).masked_fill(~m, -1e9).softmax(-1)  # formula 2: additive attention
        return (a.unsqueeze(-1) * h).sum(1)  # (b, DIM)
def loss_fn(enc, x, pos, uid):
    q = enc.query(x); e = enc.emb(pos); s = q @ e.T  # (b, b): diagonal = positive
    s = s - LAM * log_q[pos][None, :]; s = s + torch.diag(LAM * log_q[pos])  # logQ on negatives only
    fn = (pos[None, :] == pos[:, None]) | (uid[None, :] == uid[:, None]); fn.fill_diagonal_(False)
    return F.cross_entropy(s.masked_fill(fn, -1e12), torch.arange(len(pos)))
@torch.no_grad()
def evaluate(enc, hist, targ):
    enc.eval(); x = pad([r[1:] for r in hist]); scores = enc.query(x) @ enc.emb.weight.T; scores[:, PAD] = -1e9; scores[:, MASK] = -1e9
    top = scores.topk(1000, dim=1).indices.numpy(); nd, rc = [], []
    for i, r in enumerate(targ):
        t = set(r[1:]); hits = [1.0 if j in t else 0.0 for j in top[i, :20]]
        dcg = sum(h / math.log2(k + 2) for k, h in enumerate(hits)); idcg = sum(1 / math.log2(k + 2) for k in range(min(20, len(t))))
        nd.append(dcg / idcg); rc.append(sum(1 for j in top[i] if j in t) / len(r[1:]))
    enc.train(); return float(np.mean(nd)), float(np.mean(rc))
enc = Enc(); opt = torch.optim.Adam(enc.parameters(), lr=1e-3); best = (-1, None, None)
for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(len(ladder)); tot = 0.0
    for b in range(0, len(perm) - B + 1, B):
        idx = perm[b:b + B]; loss = loss_fn(enc, X[idx], targets[idx], users[idx]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    v = evaluate(enc, vh, vt); print(f'ep {ep:2d} loss {tot / (len(perm) // B):.3f} val ndcg@20 {v[0]:.4f} recall@1000 {v[1]:.4f}', flush=True)
    if v[0] > best[0]: best = (v[0], ep, {k: t.clone() for k, t in enc.state_dict().items()})
enc.load_state_dict(best[2]); t = evaluate(enc, th, tt)
print(f'RESULT lambda={LAM} seed={SEED}: best val ndcg@20 {best[0]:.4f} @ep{best[1]} -> TEST ndcg@20 {t[0]:.4f} recall@1000 {t[1]:.4f}')
