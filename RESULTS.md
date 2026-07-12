# Experiment results: logQ correction study on MCLSR (Amazon Clothing)

All numbers are best `ndcg@20` over training (validation / test users), single seed 42,
same GPU node, full-catalog ranking, standard NDCG normalization (a ComiRec-style
`comirec-ndcg` metric is available for comparison with the MCLSR paper's tables).
Measured run-to-run noise floor: **±0.0005–0.001** — differences within it are ties.
Configs live in `configs/train/grid/`, one question per config.

## 1. Retrieval loss (L_P): the correction works and is exact

| config | val | test | note |
|---|---|---|---|
| 01 in-batch, no correction | 0.0168 | 0.0182 | biased baseline |
| **02 in-batch + logQ** | 0.0245 | **0.0236** | **+30%** |
| 02 + leave-own-out q' | 0.0253 | 0.0230 | = 02 (q' is a ≤1.6e-3 row constant here) |
| **14 exact full softmax** (no sampling) | 0.0246 | **0.0234** | gold standard |

**In-batch + logQ closes 100% of the gap to the exact full softmax** (0.0236 ≈ 0.0234).

Same effect on a second architecture (SASRec, only λ changes):

| config | val | test |
|---|---|---|
| SASRec BCE baseline | 0.0138 | 0.0123 |
| SASRec in-batch, λ=0 | 0.0110 | 0.0085 |
| **SASRec in-batch, λ=1** | **0.0208** | **0.0168** |

Cosine scoring on the retrieval loss (with logQ) degrades monotonically as the score
scale shrinks relative to the fixed correction (dot → τ=0.1 → τ=0.5 → τ=1):
test 0.0236 → 0.0231 → 0.0169 → 0.0153. Keep dot products on retrieval.

## 2. Interest-level contrastive L_IL: the correction hurts — anatomy of why

Base = 03 (graph + L_IL + downstream logQ). λ sweep shows a dose response; every
"distribution fix" changes nothing; removing the *margin* restores the dynamics.

| L_IL variant | val | test | peak step |
|---|---|---|---|
| **03: no correction (λ=0)** | 0.0276 | **0.0279** | ~25k |
| 04_l00 (fps_logq λ=0, masked) | 0.0276 | 0.0283 | ~25k |
| 04 λ=0.25 | 0.0270 | 0.0265 | |
| 04 λ=0.5 | 0.0267 | 0.0258 | |
| 04 λ=1 (two runs) | 0.0264 / 0.0269 | 0.0265 / 0.0262 | **~57k** |
| 04 λ=1 + leave-own-out q' | 0.0255 | 0.0251 | ~57k |
| 04 λ=1, no masking | 0.0269 | 0.0263 | ~57k |
| 04 λ=1 **centered** (margin removed) | 0.0286 | 0.0269 | **~25k** |
| 04 λ=1 **positive also corrected** | 0.0289 | 0.0266 | ~25k |
| 04 λ=1 + **cosine** similarity | 0.0295 | 0.0271 | ~13k |

Mechanism: with the negatives-only convention the common part of the correction
(~5.6 nats, vs ±0.4 of actual per-user reweighting) becomes an uncancelled margin.
Under unbounded dot scores the model fights it by inflating norms → slow rise, late
peak, −0.002 test. Removing the margin (centering ≡ correcting the positive) or
bounding the scores (cosine) each independently restores the dynamics — but never
beats "no correction": **alignment losses have no corpus-target distribution to
debias.**

Sanity variants: B×B cross-view-only scheme (one negative per user) = 2B×2B within
noise (0.0268/0.0283 vs 0.0279/0.0276); shared L_IL projection head (paper eq. 7)
= separate heads within noise (0.0271 vs 0.0279); cosine similarity (paper eq. 8)
= dot on quality, 2× faster convergence (peak ~12k vs ~25k).

## 3. Feature-level contrastive L_UC / L_IC: nothing to correct — proven directly

Isolated pairs (downstream logQ + one feature branch), plus an **exact full-softmax
anchor**: batch anchors scored against ALL users/items (no sampling at all).

| variant | val | test |
|---|---|---|
| 09 L_IC λ=0 | 0.0244 | 0.0229 |
| 10 L_IC λ=1 | 0.0243 | 0.0239 |
| 09/10 under cosine | 0.0235 / 0.0243 | 0.0223 / 0.0233 |
| **16 L_IC exact full softmax** | 0.0239 | 0.0233 |
| 11 L_UC λ=0 | 0.0243 | 0.0223 |
| 12 L_UC λ=1 + q' | 0.0242 | 0.0225 |
| 13 L_UC λ=1, no masking | 0.0230 | 0.0228 |
| **15 L_UC exact full softmax** | 0.0236 | 0.0226 |

**Even the exact full softmax matches in-batch** on both feature branches:
in-batch sampling loses nothing here, so there is nothing for a correction to fix.

## 4. Full model

| config | val | test |
|---|---|---|
| 05 graph + all contrastive, no logQ on them | 0.0273 | 0.0276 |
| 06 + logQ on L_UC/L_IC | 0.0288 | 0.0258 |
| 07 + logQ on everything | 0.0276 | 0.0248 |

Best overall recipe remains **03: graph + logQ on the retrieval loss only** (0.0279).

## 5. Reproducibility

Same node + same seed reproduces to 4 decimals (03 rerun exact). Concurrent twins
with `deterministic: true` still diverge from step ~130 (graph CUDA ops have no
deterministic implementation) up to ~0.001 ndcg@20 — hence the noise floor above,
paired arms always run on one node, and final tables should use 3 seeds.

## The recipe

| loss type | correction |
|---|---|
| retrieval (sampled/in-batch softmax over a catalog) | logQ on **negatives only**; leave-own-out q' only matters when one id holds a visible share of the data |
| contrastive alignment (SimCLR-style views) | **no correction**; if one is ever applied, use a margin-free form (centered / positive-corrected) and cosine scoring |
