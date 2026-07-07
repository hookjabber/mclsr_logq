"""Reference tests for the logQ-corrected losses.

Each loss is checked against an independent naive (loop-based) implementation
of the intended math. Run directly:

    python tests/test_logq_losses.py

or via pytest.
"""
import os
import pickle
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from irec.loss.base import FpsLogQLoss, MCLSRLogqInBatchLoss  # noqa: E402

B = 8
D = 16
COUNTS = [1.0, 500.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 5.0]
POS_IDS = torch.tensor([1, 1, 2, 3, 4, 5, 6, 7])           # duplicate item id=1
USER_IDS = torch.tensor([10, 11, 12, 12, 13, 14, 15, 16])  # duplicate user id=12
FPS_IDS = torch.tensor([1, 1, 2, 3, 4, 5, 6, 7])


def _counts_path():
    f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump(torch.tensor(COUNTS).numpy(), f)
    f.close()
    return f.name


def _probs():
    c = torch.tensor(COUNTS, dtype=torch.float64)
    return torch.clamp(c / c.sum(), min=1e-10)


def reference_inbatch(q, p_emb, lam, loo):
    prob = _probs()
    s = q.detach().double() @ p_emb.double().T
    for i in range(B):
        for j in range(B):
            if i == j:
                continue  # positive: never corrected
            corr = torch.log(prob[POS_IDS[j]])
            if loo:
                corr = corr - torch.log(torch.clamp(1 - prob[POS_IDS[i]], min=1e-10))
            s[i, j] = s[i, j] - lam * corr
    for i in range(B):
        for j in range(B):
            if i != j and (POS_IDS[i] == POS_IDS[j] or USER_IDS[i] == USER_IDS[j]):
                s[i, j] = -1e12
    return torch.nn.functional.cross_entropy(s.float(), torch.arange(B))


def reference_fps(fst, snd, tau, lam, mask_fn, loo, n_draws):
    prob = _probs()
    z = torch.cat((fst.detach(), snd), 0).double()
    scores = (z @ z.T) / tau
    cid = torch.cat((FPS_IDS, FPS_IDS))
    V = 2 * B
    corr = torch.zeros(V, V, dtype=torch.float64)
    for a in range(V):
        for j in range(V):
            pj = prob[cid[j]]
            if loo:
                pj = torch.clamp(pj / torch.clamp(1 - prob[cid[a]], min=1e-10), min=1e-10, max=1.0)
            qq = 1.0 - (1.0 - pj) ** n_draws
            corr[a, j] = torch.log(torch.clamp(qq, min=1e-10))
    scores = scores - lam * corr
    pos = [(i + B) % V for i in range(V)]
    for a in range(V):
        scores[a, pos[a]] += lam * corr[a, pos[a]]
    for a in range(V):
        for j in range(V):
            if a == j:
                scores[a, j] = -1e12
            elif mask_fn and cid[a] == cid[j] and j != pos[a]:
                scores[a, j] = -1e12
    return torch.nn.functional.cross_entropy(scores.float(), torch.tensor(pos)) / 2


def test_inbatch_matches_reference():
    torch.manual_seed(0)
    path = _counts_path()
    q = torch.randn(B, D, requires_grad=True)
    p_emb = torch.randn(B, D)
    for lam, loo in [(1.0, False), (1.0, True), (0.0, False), (0.0, True)]:
        loss = MCLSRLogqInBatchLoss(
            queries_prefix='q', positive_prefix='p', positive_ids_prefix='pid',
            path_to_item_counts=path, logq_lambda=lam,
            leave_own_out=loo, user_ids_prefix='uid',
        )
        got = loss({'q': q, 'p': p_emb, 'pid': POS_IDS, 'uid': USER_IDS})
        want = reference_inbatch(q, p_emb, lam, loo)
        assert torch.allclose(got, want, atol=1e-4), (lam, loo, got.item(), want.item())
    got.backward()
    assert torch.isfinite(q.grad).all()


def test_fps_logq_matches_reference():
    torch.manual_seed(1)
    path = _counts_path()
    fst = torch.randn(B, D, requires_grad=True)
    snd = torch.randn(B, D)
    cases = [
        (1.0, True, False),   # plain vector correction, masked
        (1.0, True, True),    # leave-own-out (q')
        (1.0, False, False),  # no masking, plain q
        (0.0, True, False),   # lambda=0
    ]
    for lam, mask_fn, loo in cases:
        loss = FpsLogQLoss(
            fst_embeddings_prefix='f', snd_embeddings_prefix='s', ids_prefix='i',
            path_to_counts=path, tau=0.5, logq_lambda=lam,
            logq_probability_mode='inbatch_negative',
            mask_false_negatives=mask_fn, leave_own_out=loo,
        )
        got = loss({'f': fst, 's': snd, 'i': FPS_IDS})
        want = reference_fps(fst, snd, 0.5, lam, mask_fn, loo, B - 1)
        assert torch.allclose(got, want, atol=1e-4), (lam, mask_fn, loo, got.item(), want.item())
    got.backward()
    assert torch.isfinite(fst.grad).all()


def test_fps_logq_lambda_zero_invariance():
    torch.manual_seed(2)
    path = _counts_path()
    fst, snd = torch.randn(B, D), torch.randn(B, D)
    vals = []
    for loo in [False, True]:
        loss = FpsLogQLoss(
            fst_embeddings_prefix='f', snd_embeddings_prefix='s', ids_prefix='i',
            path_to_counts=path, tau=0.5, logq_lambda=0.0,
            mask_false_negatives=True, leave_own_out=loo,
        )
        vals.append(loss({'f': fst, 's': snd, 'i': FPS_IDS}))
    assert torch.allclose(vals[0], vals[1], atol=1e-7)


def test_leave_own_out_requires_masking():
    path = _counts_path()
    try:
        FpsLogQLoss(
            fst_embeddings_prefix='f', snd_embeddings_prefix='s', ids_prefix='i',
            path_to_counts=path, tau=0.5,
            mask_false_negatives=False, leave_own_out=True,
        )
    except ValueError:
        return
    raise AssertionError('expected ValueError')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
    print('\nALL TESTS PASSED')
