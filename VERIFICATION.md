# Verification: what was checked, and how to re-run every check

Every claim in [RESULTS.md](./RESULTS.md) rests on the pipeline below. Each stage has an
independent check that does not reuse the framework code it verifies; all of them run on a CPU.

| stage | check | how to run | status |
|---|---|---|---|
| raw data → split files | `scripts/prepare_dataset.py` rebuilds the files from the raw Amazon ratings csv (iterative 5-core, dense ids, chronological order, users shuffled with seed 42 and split 8:1:1, train users cropped to 20 items and unrolled into a prefix ladder, eval users split at `int(0.8·len)`) and diffs them against the files used in the study | `python scripts/prepare_dataset.py --csv data/Clothing/data.csv --out /tmp/clothing --verify-against data/Clothing` | Clothing: all six files identical to the study files; the 5-core statistics equal Table 1 of the MCLSR paper (39 387 users, 23 033 items, 278 677 actions). Beauty: identical from the 5-core sequence file (`--seq`). CDs & Vinyl: the user split equals the seed-42 protocol over the user list, the ladder equals the prefixes of the cropped sequences |
| split files → graphs | `scripts/verify_graphs.py` recomputes the three graphs with scipy from the ladder: bipartite adjacency and D^-½ A D^-½ (eq. 1), user–user / item–item co-action weights R Rᵀ / Rᵀ R, top-50 per row, normalisation of the kept weights; compares with the cached `.npz` graphs | `python scripts/verify_graphs.py Clothing Beauty Toys` | all three datasets: max deviation 0, no edge outside the co-action pairs, every row keeps min(50, #neighbours) strongest edges |
| graphs + files + weights → model outputs | `scripts/indep_check_architecture.py` reimplements the forward pass in numpy/float64 from the model's `state_dict` and the raw files (eq. 2–7, 9–10, 13 of the paper, the eval top-k of eq. 16), the in-batch retrieval loss with and without the logQ correction (eq. 14), the contrastive losses in the paper's form (eq. 8 / 11, cosine) and in the symmetric form used by default, and the composite objective (eq. 15); compares with the framework on a real training batch | `PYTHONPATH=src python scripts/indep_check_architecture.py --config configs/train/clothing64/05_full_l00.json --data data/Clothing` | 29/29 on Clothing, Beauty and CDs & Vinyl; largest deviation 1.2e-6 (losses), 8e-9 (representations) |
| the SASRec branch | `scripts/indep_check_sasrec.py` re-implements the causal transformer encoder in numpy (post-LN multi-head attention, feed-forward, residuals) from the saved weights; checks that the SASRec dataset feeds the evaluation exactly the same users, histories and targets as the MCLSR dataset, the per-position outputs, the eval top-k from the last position, causality (a perturbed item changes nothing before its position), padding invariance, the training outputs (one query per position, next-item positives, owner ids) and the in-batch loss on them | `PYTHONPATH=src python scripts/indep_check_sasrec.py --config configs/train/clothing64/sasrec_inbatch_l00.json --mclsr-config configs/train/clothing64/01_orig.json` | 12/12; largest deviation 9e-7 |
| hardware reproducibility | the same config and seed trained on two GPU models (RTX 3080, GTX 1080): the trajectories part from the first update, the validation curves differ by 0.0007 on average, the test metric by 0.0028 ndcg@20 — seed-noise size; all comparisons are therefore made within one GPU model | `configs/train/beauty/05_full_l00_gpucheck.json` vs `05_full_l00`, seed 1; `scripts/val_test_curves.py` for the curves | documented in RESULTS.md A.0 |
| gradient flow | each loss term trains exactly the parameter groups the equations connect it to: the retrieval loss reaches embeddings, positions, attention and eq. 5 but not the projectors; L_IL reaches both eq. 7 projectors; L_UC / L_IC reach only their projector and, through an even number of propagation steps, only the embedding table of their own entity type | part of the audit log, reproduced by `scripts/indep_check_architecture.py` style code | as expected |
| loss and metric reference implementations | `tests/test_logq_losses.py`, `tests/test_explicit_negative_losses.py`, `tests/test_metrics.py`, `tests/test_graph.py` compare every loss variant (three forms of the correction, mixed negative sampling, exact full softmax, the paper scheme, euclidean scoring) and every metric with hand-written implementations | `python -m pytest -q tests` | 42 tests |
| the whole pipeline on synthetic data | `tests/test_end_to_end.py` writes a small synthetic dataset in the study's file formats, runs the confirmatory runner (graphs built from the files, full model with all four losses and the correction, validation-selected checkpoint, one test read, report with content hashes), checks the report, and repeats the run with the same seed: the test numbers must be identical | `python -m pytest -q tests/test_end_to_end.py` | passes in ~4 s on a CPU |
| published numbers from checkpoints | `scripts/eval_checkpoint.py` scores any saved checkpoint; `scripts/indep_check_beauty.py` and `scripts/indep_check_clothing.py` train the retrieval model without the framework and reproduce the direction and size of the headline effect | see the script docstrings | reproduced |
| uncertainty of the headline comparisons | `scripts/paired_bootstrap.py` scores the validation-selected checkpoints of an arm and its baseline on the same test users, pools the per-user differences over seeds, and reports a user-bootstrap 95 % interval and a sign-flip permutation p-value | `PYTHONPATH=src python scripts/paired_bootstrap.py --prefix beauty --config-dir configs/train/beauty --pairs 02_logq_downstream:01_orig … --output results/stats/beauty_paired_bootstrap.md` | [results/stats/](./results/stats/) |

## Paper ↔ code map

MCLSR (Wang et al., CIKM 2022), equations as numbered in the paper; the framework code in `src/irec/`.

| eq. | paper | code | deviation |
|---|---|---|---|
| §3.2 | M_uv weights = interaction counts; M_uu = M_uv M_uvᵀ, M_vv likewise; 50 neighbours per node | `dataset/graph.py` `GraphDataset`: binary (user, item) pairs, co-action counts, top-50 per row | binary instead of counted bipartite edges (5-core Amazon has almost no repeats) |
| 1 | X(l) = D^-½ A D^-½ X(l−1) | `GraphDataset.get_sparse_graph_layer`, `MCLSRModel._apply_graph_encoder` (last layer; LightGCN layer mean behind `graph_layer_mean`) | edge dropout 0.3 during training (A/B: no gain without it) |
| 2–3 | A_s = softmax(W₂ tanh(W₁ E_{u,p}ᵀ)), I_s = A_s E_u | `models/mclsr.py` forward, `_current_interest_learning_encoder` | LayerNorm and dropout 0.3 on E_{u,p}; positions counted from the end |
| 4–6 | H_all,uv = GraphEncoder([H_u ‖ H_v], G_uv); A_c = softmax(tanh(W₃ h_u,uv) E_u,uvᵀ); I_c = A_c E_u,uv | forward, `_general_interest_learning_encoder` | none |
| 7 | T = W₂ᵖ ELU(W₁ᵖ I + b₁ᵖ) + b₂ᵖ, shared for both views | `_sequential_projector`, `_graph_projector`; shared with `share_il_projector: true` | two projectors by default (A/B: no difference) |
| 8, 11 | L_IL / L_UC: anchor = one view, candidates = all rows of the other view plus the other rows of the anchor view, cosine, τ = 0.5 | `loss/contrastive.py` `FpsLoss`: `scheme: paper` + `normalize_embeddings: true`; default `symmetric` scheme with dot products | default is the SimCLR-style symmetric form with dot products; its loss is half the per-anchor cross-entropy, so the effective β and γ are half the configured values (β plateau 0.25–1 makes this immaterial); the paper form is run as the `*_paper` arms |
| 9–10, 12 | H_u,uu = GraphEncoder(H_u, G_uu), shared MLP per entity type, L_FL = L_UC + L_IC | forward, `_user_projection`, `_item_projection`; L_IC over the unique items of the batch | none |
| 13 | I_comb = α I_s + (1−α) I_c, α = 0.5 | forward | none |
| 14 | sampled softmax on I_comb with 1280 negatives | `loss/retrieval.py` `InBatchLogQLoss`: in-batch negatives (127), same-item and same-user masks, three forms of the logQ correction; `01_uniform1280` arm = 1280 uniform negatives; `FullSoftmaxLoss` = exact softmax | the study's object: which sampler and which correction |
| 15 | J = L_P + β L_IL + γ L_FL, β = 1, γ = 0.05 | `CompositeLoss`, config weights 1 / 1 / 0.05 / 0.05 | see eq. 8 note |
| 16 | top-N by (I_s)ᵀ h_v over the catalogue | forward in eval mode, `eval_top_k` | none |
| §5.1 | 5-core, train length 20, users 8:1:1, every item a target, inference from the first 80 % | `scripts/prepare_dataset.py`, `dataset/sequences.py` `MCLSRDataset` | none |

## The three forms of the correction, and the released implementation

For a query with positive item p, in-batch negatives N (the positives of the other rows, minus the
rows with the same item and, when configured, the same user), scores s and the sampling
distribution q(v) = count(v) / Σ count (`item_counts.pkl`, training frequencies), the retrieval loss
`mclsr_logq_inbatch` has these forms (`src/irec/loss/retrieval.py`):

| form | config | per-query loss |
|---|---|---|
| in-batch, no correction | `logq_lambda: 0` | −s_p + log(e^{s_p} + Σ_{j∈N} e^{s_j}) |
| negatives-only (the study's default) | `variant: negatives_only`, λ | −s_p + log(e^{s_p} + Σ_j e^{s_j − λ log q(j)}) |
| standard (Bengio & Senécal; Yi et al. 2019) | `variant: standard` | −(s_p − λ log q(p)) + log(e^{s_p − λ log q(p)} + Σ_j e^{s_j − λ log q(j)}) |
| corrected (Khrylchenko et al., RecSys 2025) | `variant: corrected` | sg(1 − P̂) · (−s_p + log Σ_j e^{s_j − λ log Q′(j)}), Q′(j) = q(j) / (1 − q(p)), P̂ = e^{s_p} / (e^{s_p} + mean_j e^{s_j − λ log Q′(j)}) |
| leave-own-out q′ | `leave_own_out: true` | negatives-only with q′(j) = q(j) / (1 − q(p)) |
| mixed negative sampling | `mixed_uniform_negatives: K` | K shared uniform items added to the pool, correction with the mixture proposal B/(B+K)·q + K/(B+K)/|V| |

Every form has a hand-written reference in `tests/test_logq_losses.py`. In addition, the
negatives-only and corrected forms were compared on one batch with the loss lines of the training
scripts released with the RecSys 2025 paper (`train_in_batch_logq_old.py`, `train_in_batch_logq_new.py`
of the authors' repository), with the same in-batch pool: values agree to 1e-6 and gradients to
1e-9 at three score scales. Two remarks on the released scripts: their "new" script also mixes
uniform negatives into the pool (half of it) and applies the same frequency correction to them,
whereas the `mixed_uniform_negatives` arms here use the mixture proposal; and their
`clamp(min=1e-10)` on the softmax ratio caps a per-query loss at 23 nats, which is inactive in
training (losses of 5–8 nats) but makes a comparison on unscaled random inputs meaningless.
