#!/usr/bin/env python
"""
Independent replication of the logQ-correction effect on Amazon Clothing, written without
reading the framework code (src/, scripts/): own data loading, own encoder (MCLSR paper eq. 2-3),
own in-batch softmax with the negatives-only correction, own metrics.

    python scripts/indep_check_clothing.py --lam 0 --seed 0 --epochs 8 --out results/indep/clothing_lam0_seed0.json
    python scripts/indep_check_clothing.py --lam 1 --seed 0 --epochs 8 --out results/indep/clothing_lam1_seed0.json

Result (3 seeds, 8 epochs, CPU): lam=0 -> 0.0143 / 0.203, lam=1 -> 0.0211 / 0.288 (test ndcg@20 / recall@1000),
+47 % / +42 %, positive on every seed; the framework reports 0.0152 / 0.252 -> 0.0226 / 0.314.

Model : item emb (d=64) + reversed learned position emb -> LayerNorm -> dropout(0.3)
        -> additive attention  a = softmax_t( w2^T tanh(W1 x_t) )
        -> I_s = sum_t a_t * e_t   (e_t = RAW item embedding, paper eq. 3)
Score : s(u, j) = I_s . e_j
Loss  : in-batch sampled softmax over the ladder (prefix -> target) pairs.
        S = I_s @ E[targets]^T ; diagonal = positive.
        Off-diagonal entries with the same target item as the row, or the same
        user as the row, are masked to -1e9.
        logQ arm (lam=1): S[i, j] -= lam * log q(target_j)  for all j != i,
        where q = train-frequency of the item (from train_sasrec.txt).
Eval  : full-catalogue ranking over items 1..N, NDCG@20 (IDCG over min(20,|T|))
        and Recall@1000 (= hits@1000 / |T|), averaged over users.
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = 'data/Clothing'
MAXLEN = 20


# ----------------------------------------------------------------------------- data
def read_lines(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            rows.append((int(p[0]), [int(x) for x in p[1:]]))
    return rows


def pad_left(seqs, maxlen=MAXLEN):
    """Keep the most recent `maxlen` items, left-pad with 0 (most recent item in the last slot)."""
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:]
        if s:
            out[i, maxlen - len(s) :] = s
    return out


def load_all():
    train_full = read_lines(os.path.join(DATA, 'train_sasrec.txt'))
    ladder = read_lines(os.path.join(DATA, 'train_mclsr.txt'))
    vh = read_lines(os.path.join(DATA, 'valid_history.txt'))
    vt = read_lines(os.path.join(DATA, 'valid_target.txt'))
    th = read_lines(os.path.join(DATA, 'test_history.txt'))
    tt = read_lines(os.path.join(DATA, 'test_target.txt'))

    n_items = 0
    for rows in (train_full, ladder, vh, vt, th, tt):
        for _, items in rows:
            n_items = max(n_items, max(items))

    # item frequency q(j) from ALL occurrences in train_sasrec.txt
    counts = np.zeros(n_items + 1, dtype=np.float64)
    for _, items in train_full:
        for it in items:
            counts[it] += 1
    total = counts.sum()
    q = counts / total
    log_q = np.full(n_items + 1, math.log(1.0 / total))  # floor for unseen items (never used as negatives)
    log_q[counts > 0] = np.log(q[counts > 0])

    # training ladder: last item on the line = target, everything before = input prefix
    lad_user = np.array([u for u, _ in ladder], dtype=np.int64)
    lad_tgt = np.array([items[-1] for _, items in ladder], dtype=np.int64)
    lad_seq = pad_left([items[:-1] for _, items in ladder])
    assert (lad_seq.sum(1) > 0).all(), 'empty prefix in ladder'

    def eval_set(hist_rows, tgt_rows):
        hmap = {u: s for u, s in hist_rows}
        tmap = {u: s for u, s in tgt_rows}
        users = sorted(set(hmap) & set(tmap))
        assert len(users) == len(hmap) == len(tmap), 'history/target user mismatch'
        seq = pad_left([hmap[u] for u in users])
        tg = [tmap[u] for u in users]
        return seq, tg

    val = eval_set(vh, vt)
    test = eval_set(th, tt)

    # sanity: users disjoint across splits
    tr_users = {u for u, _ in train_full}
    va_users = {u for u, _ in vh}
    te_users = {u for u, _ in th}
    assert not (tr_users & va_users) and not (tr_users & te_users) and not (va_users & te_users)

    info = dict(
        n_items=n_items,
        n_train_users=len(train_full),
        n_ladder=len(ladder),
        n_val_users=len(val[1]),
        n_test_users=len(test[1]),
        train_tokens=int(total),
        items_with_zero_train_count=int((counts[1:] == 0).sum()),
    )
    return n_items, torch.tensor(log_q, dtype=torch.float32), (lad_seq, lad_user, lad_tgt), val, test, info


# ----------------------------------------------------------------------------- model
class CurrentInterest(nn.Module):
    def __init__(self, n_items, dim=64, maxlen=MAXLEN, dropout=0.3):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(maxlen, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.W1 = nn.Linear(dim, 4 * dim, bias=False)  # W1 in R^{4d x d}
        self.w2 = nn.Linear(4 * dim, 1, bias=False)  # W2 in R^{4d}
        nn.init.normal_(self.item.weight, std=0.1)
        with torch.no_grad():
            self.item.weight[0].zero_()
        nn.init.normal_(self.pos.weight, std=0.1)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        # reversed positions: the most recent item (last slot) gets position 0
        self.register_buffer('pos_ids', torch.arange(maxlen - 1, -1, -1))

    def forward(self, seq):  # seq: [B, T] left-padded, 0 = pad
        mask = seq > 0
        e = self.item(seq)  # raw item embeddings  [B, T, d]
        x = e + self.pos(self.pos_ids)[None]  # + position           (paper: E_{u,p})
        x = self.drop(self.ln(x))
        att = self.w2(torch.tanh(self.W1(x))).squeeze(-1)  # [B, T]
        att = att.masked_fill(~mask, -1e9)
        a = torch.softmax(att, dim=-1)  # eq. (2)
        return (a.unsqueeze(-1) * e).sum(1)  # eq. (3): I_s = A_s E_u


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, seq_np, targets, n_items, exclude_hist=False, bs=1024):
    model.eval()
    E = model.item.weight[1:]  # items 1..N  -> row j-1
    disc = 1.0 / torch.log2(torch.arange(2, 22, dtype=torch.float32))  # rank 1..20
    idcg_table = torch.cumsum(disc, 0)  # idcg for |T| = 1..20
    ndcg_sum, rec_sum, n = 0.0, 0.0, 0
    for b in range(0, len(seq_np), bs):
        seq = torch.from_numpy(seq_np[b : b + bs])
        interest = model(seq)
        scores = interest @ E.T  # [B, N]
        if exclude_hist:  # secondary metric: history items removed from the ranking
            valid = seq > 0
            rows = torch.arange(len(seq))[:, None].expand_as(seq)
            scores[rows[valid], seq[valid] - 1] = -1e9
        top = scores.topk(1000, dim=1).indices + 1  # item ids, [B, 1000]
        tg = targets[b : b + bs]
        dense = torch.zeros(len(tg), n_items + 1, dtype=torch.bool)
        for r, t in enumerate(tg):
            dense[r, t] = True
        hit = dense.gather(1, top).float()  # [B, 1000]
        nt = torch.tensor([len(set(t)) for t in tg], dtype=torch.float32)
        dcg = (hit[:, :20] * disc).sum(1)
        idcg = idcg_table[(nt.clamp(max=20) - 1).long()]
        ndcg_sum += float((dcg / idcg).sum())
        rec_sum += float((hit.sum(1) / nt).sum())
        n += len(tg)
    model.train()
    return ndcg_sum / n, rec_sum / n


# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--max_steps', type=int, default=0, help='debug: stop after this many steps')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    t0 = time.time()
    n_items, log_q, (lad_seq, lad_user, lad_tgt), (vseq, vtg), (tseq, ttg), info = load_all()
    print('data:', json.dumps(info), f'({time.time() - t0:.1f}s)', flush=True)

    model = CurrentInterest(n_items)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    L = len(lad_tgt)
    B = args.batch
    steps_per_epoch = math.ceil(L / B)
    log_q_t = log_q

    lad_seq_t = torch.from_numpy(lad_seq)
    lad_user_t = torch.from_numpy(lad_user)
    lad_tgt_t = torch.from_numpy(lad_tgt)

    history = []
    step = 0
    t_train = 0.0
    for ep in range(1, args.epochs + 1):
        perm = torch.from_numpy(rng.permutation(L))
        model.train()
        te = time.time()
        loss_sum, nb = 0.0, 0
        for b in range(0, L, B):
            idx = perm[b : b + B]
            seq, usr, tgt = lad_seq_t[idx], lad_user_t[idx], lad_tgt_t[idx]
            n = len(idx)
            interest = model(seq)  # [n, d]
            Et = model.item(tgt)  # [n, d]
            S = interest @ Et.T  # [n, n], diagonal = positive
            eye = torch.eye(n, dtype=torch.bool)
            if args.lam != 0.0:
                corr = args.lam * log_q_t[tgt][None, :]  # per-column log q(target_j)
                S = S - corr.masked_fill(eye, 0.0)  # positive (diagonal) left unchanged
            same_t = tgt[:, None] == tgt[None, :]
            same_u = usr[:, None] == usr[None, :]
            S = S.masked_fill((same_t | same_u) & ~eye, -1e9)
            loss = F.cross_entropy(S, torch.arange(n))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss)
            nb += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        t_ep = time.time() - te
        t_train += t_ep
        tv = time.time()
        v_ndcg, v_rec = evaluate(model, vseq, vtg, n_items)
        s_ndcg, s_rec = evaluate(model, tseq, ttg, n_items)
        s_ndcg_x, s_rec_x = evaluate(model, tseq, ttg, n_items, exclude_hist=True)
        t_ev = time.time() - tv
        rec = dict(
            epoch=ep,
            steps=step,
            train_loss=loss_sum / max(nb, 1),
            val_ndcg20=v_ndcg,
            val_recall1000=v_rec,
            test_ndcg20=s_ndcg,
            test_recall1000=s_rec,
            test_ndcg20_exclhist=s_ndcg_x,
            test_recall1000_exclhist=s_rec_x,
            epoch_train_sec=t_ep,
            eval_sec=t_ev,
        )
        history.append(rec)
        print(
            f'lam={args.lam} seed={args.seed} ep {ep:3d} step {step:6d} loss {rec["train_loss"]:.4f} | '
            f'val ndcg@20 {v_ndcg:.5f} rec@1000 {v_rec:.4f} | test ndcg@20 {s_ndcg:.5f} rec@1000 {s_rec:.4f} '
            f'(excl-hist {s_ndcg_x:.5f}/{s_rec_x:.4f}) | {t_ep:.1f}s train, {t_ev:.1f}s eval',
            flush=True,
        )
        if args.max_steps and step >= args.max_steps:
            break

    best = max(history, key=lambda r: r['val_ndcg20'])
    result = dict(
        args=vars(args),
        info=info,
        steps_per_epoch=steps_per_epoch,
        history=history,
        best_epoch=best['epoch'],
        best=best,
        total_train_sec=t_train,
        wall_sec=time.time() - t0,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(
        f'BEST (by val ndcg@20) lam={args.lam} seed={args.seed}: epoch {best["epoch"]} '
        f'val ndcg@20 {best["val_ndcg20"]:.5f} | test ndcg@20 {best["test_ndcg20"]:.5f} '
        f'recall@1000 {best["test_recall1000"]:.4f} | wall {time.time() - t0:.0f}s',
        flush=True,
    )


if __name__ == '__main__':
    main()
