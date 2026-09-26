# Run index — every training run of the logQ study, labelled

One row per arm and dataset. **Seeds** columns: confirmatory runs (`scripts/train_confirmatory.py`, test read once on the validation-selected checkpoint; reports in `results/confirm/`, mean ± std over the listed seeds). **Single run** columns: exploratory run with seed 42 (test taken at the validation-best step from the TensorBoard log; approximate to ±1 evaluation interval). Metrics: test ndcg@20 / recall@1000 over the full catalogue. Figures and headline tables: `RESULTS.md`.

## Clothing — 52 arms

| arm (config) | what it is | seeds: ndcg@20 / recall@1000 | seeds | single run (seed 42): ndcg@20 / recall@1000 | runs |
|---|---|---|---|---|---|
| [01_orig](configs/train/clothing64/01_orig.json) | MCLSR without graph; in-batch sampled softmax, no correction (λ=0) | 0.0152 ± 0.0004 / 0.2522 ± 0.0054 | 1,2,3 | 0.0163 / 0.2421 | 8 |
| [02_cosine_t01](configs/train/grid/02_cosine_t01.json) | 02 with cosine scores on L_P, τ=0.1 | — | — | 0.0228 / 0.3223 | 1 |
| [02_cosine_t05](configs/train/grid/02_cosine_t05.json) | 02 with cosine scores on L_P, τ=0.5 | — | — | 0.0159 / 0.2586 | 1 |
| [02_cosine_t1](configs/train/grid/02_cosine_t1.json) | 02 with cosine scores on L_P, τ=1 | — | — | 0.0152 / 0.2579 | 1 |
| [02_logq_downstream](configs/train/clothing64/02_logq_downstream.json) | MCLSR without graph; in-batch + logQ on L_P (λ=1) | 0.0226 ± 0.0008 / 0.3143 ± 0.0014 | 1,2,3 | 0.0230 / 0.3113 | 4 |
| [02_logq_downstream_loo](configs/train/grid/02_logq_downstream_loo.json) | 02 with leave-own-out q′ | — | — | 0.0223 / 0.3128 | 1 |
| [02_logq_targetq](configs/train/grid/02_logq_targetq.json) | 02 with q from target counts | — | — | 0.0229 / 0.3110 | 1 |
| [03_graph](configs/train/clothing64/03_graph.json) | full model: user–item graph + L_IL; logQ on L_P | 0.0261 ± 0.0007 / 0.3581 ± 0.0024 | 1,2,3 | 0.0272 / 0.3593 | 5 |
| [03_graph_bxb](configs/train/grid/03_graph_bxb.json) | 03, L_IL over the batch×batch pool (August variant, see RESULTS.md Part B) | — | — | 0.0251 / 0.3499 | 1 |
| [03_graph_bxb_w05](configs/train/grid/03_graph_bxb_w05.json) | 03 batch×batch, γ=0.5 | — | — | 0.0285 / 0.3499 | 1 |
| [03_graph_cosine](configs/train/grid/03_graph_cosine.json) | 03 with cosine L_IL | — | — | 0.0275 / 0.3544 | 1 |
| [03_graph_cosine_t01](configs/train/grid/03_graph_cosine_t01.json) | 03 with cosine L_IL, τ=0.1 | — | — | 0.0255 / 0.3457 | 1 |
| [03_graph_cosine_t02](configs/train/grid/03_graph_cosine_t02.json) | 03 with cosine L_IL, τ=0.2 | — | — | 0.0254 / 0.3508 | 1 |
| [03_graph_cosine_t1](configs/train/grid/03_graph_cosine_t1.json) | 03 with cosine L_IL, τ=1 | — | — | 0.0254 / 0.3324 | 1 |
| [03_graph_euclid](configs/train/grid/03_graph_euclid.json) | 03 with squared-euclidean L_IL | — | — | 0.0246 / 0.3554 | 2 |
| [03_graph_euclid_t1](configs/train/grid/03_graph_euclid_t1.json) | 03 with squared-euclidean L_IL, τ=1 | — | — | 0.0264 / 0.3593 | 1 |
| [03_graph_paper_faithful](configs/train/grid/03_graph_paper_faithful.json) | 03 with shared projector, cosine L_IL and the paper scheme | — | — | 0.0250 / 0.3217 | 1 |
| [03_graph_paper_w05](configs/train/grid/03_graph_paper_w05.json) | 03 paper-faithful with γ=0.5 | — | — | 0.0260 / 0.3554 | 1 |
| [03_graph_shared_proj](configs/train/grid/03_graph_shared_proj.json) | 03 with a shared L_IL projector | — | — | 0.0255 / 0.3634 | 1 |
| [04_graph_logq_lil](configs/train/clothing64/04_graph_logq_lil.json) | as 03, logQ also on L_IL | 0.0249 ± 0.0014 / 0.3439 ± 0.0043 | 1,2,3 | 0.0262 / 0.3467 | 5 |
| 04_graph_logq_lil_centered | 04 with centred log q on L_IL | — | — | 0.0259 / 0.3562 | 1 |
| [04_graph_logq_lil_cosine](configs/train/grid/04_graph_logq_lil_cosine.json) | 04 with cosine L_IL | — | — | 0.0267 / 0.3533 | 1 |
| [04_graph_logq_lil_det](configs/train/grid/04_graph_logq_lil_det.json) | 04 with deterministic kernels | — | — | 0.0246 / 0.3448 | 2 |
| [04_graph_logq_lil_l00](configs/train/grid/04_graph_logq_lil_l00.json) | 04 with λ_IL=0 (control) | — | — | 0.0275 / 0.3591 | 1 |
| [04_graph_logq_lil_l025](configs/train/grid/04_graph_logq_lil_l025.json) | 04 with λ_IL=0.25 | — | — | 0.0250 / 0.3534 | 1 |
| [04_graph_logq_lil_l05](configs/train/grid/04_graph_logq_lil_l05.json) | 04 with λ_IL=0.5 | — | — | 0.0257 / 0.3562 | 1 |
| [04_graph_logq_lil_loo](configs/train/grid/04_graph_logq_lil_loo.json) | 04 with leave-own-out q′ on L_IL | — | — | 0.0242 / 0.3442 | 1 |
| [04_graph_logq_lil_nomask](configs/train/grid/04_graph_logq_lil_nomask.json) | 04 without false-negative masking on L_IL | — | — | 0.0251 / 0.3472 | 1 |
| 04_graph_logq_lil_pos | 04 with the positive corrected too (standard form) on L_IL | — | — | 0.0260 / 0.3533 | 1 |
| [05_full_baseline](configs/train/clothing64/05_full_baseline.json) | paper model: graph + L_IL + L_UC + L_IC (γ=0.05); logQ on L_P | — | — | 0.0266 / 0.3549 | 1 |
| [05_full_baseline_g01](configs/train/grid/05_full_baseline_g01.json) | paper model with γ=0.1 | — | — | 0.0270 / 0.3589 | 1 |
| [05_full_l00](configs/train/clothing64/05_full_l00.json) | paper model (graph + L_IL + L_UC + L_IC) without the correction (λ=0) | 0.0166 ± 0.0012 / 0.2919 ± 0.0046 | 1,2,3 | — | 3 |
| [06_full_logq_ucic](configs/train/clothing64/06_full_logq_ucic.json) | paper model, logQ also on L_UC / L_IC | — | — | 0.0253 / 0.3541 | 2 |
| [06_full_logq_ucic_g01](configs/train/grid/06_full_logq_ucic_g01.json) | paper model, logQ on L_UC / L_IC, γ=0.1 | — | — | 0.0262 / 0.3599 | 1 |
| [07_full_logq_all](configs/train/grid/07_full_logq_all.json) | paper model, logQ on every loss | — | — | 0.0129 / 0.2754 | 2 |
| [09_item_only](configs/train/clothing64/09_item_only.json) | graph + L_IC only (no L_IL, no L_UC) | — | — | 0.0221 / 0.3127 | 2 |
| [09_item_only_cosine](configs/train/grid/09_item_only_cosine.json) | graph + cosine L_IC only | — | — | 0.0220 / 0.3162 | 1 |
| [10_item_only_logq](configs/train/clothing64/10_item_only_logq.json) | graph + L_IC with logQ | — | — | 0.0213 / 0.3154 | 2 |
| [10_item_only_logq_cosine](configs/train/grid/10_item_only_logq_cosine.json) | graph + cosine L_IC with logQ | — | — | 0.0220 / 0.3182 | 1 |
| [10_item_only_logq_ctxq](configs/train/grid/10_item_only_logq_ctxq.json) | graph + L_IC with logQ from context-inclusion counts | — | — | 0.0229 / 0.3183 | 1 |
| [10_item_only_logq_ctxq_v2](configs/train/grid/10_item_only_logq_ctxq_v2.json) | graph + L_IC with logQ from context-inclusion counts | — | — | 0.0233 / 0.3226 | 1 |
| [11_user_only](configs/train/grid/11_user_only.json) | graph + L_UC only | — | — | 0.0211 / 0.3206 | 1 |
| [12_user_only_logq](configs/train/grid/12_user_only_logq.json) | graph + L_UC with logQ | — | — | 0.0224 / 0.3135 | 1 |
| [13_user_only_logq_nomask](configs/train/grid/13_user_only_logq_nomask.json) | graph + L_UC with logQ, no false-negative mask | — | — | 0.0216 / 0.3036 | 1 |
| [14_full_softmax](configs/train/clothing64/14_full_softmax.json) | exact full-catalogue softmax on L_P (no sampling), no graph | 0.0219 ± 0.0007 / 0.3106 ± 0.0016 | 1,2,3 | 0.0222 / 0.2987 | 4 |
| [15_user_only_full](configs/train/grid/15_user_only_full.json) | graph + full-catalogue L_UC only (unmatched) | — | — | 0.0212 / 0.3134 | 1 |
| [16_item_only_full](configs/train/grid/16_item_only_full.json) | graph + full-catalogue L_IC only (unmatched) | — | — | 0.0224 / 0.3211 | 1 |
| [17_user_only_full_matched](configs/train/grid/17_user_only_full_matched.json) | graph + full-catalogue matched L_UC only | — | — | 0.0210 / 0.3147 | 1 |
| [18_item_only_full_matched](configs/train/grid/18_item_only_full_matched.json) | graph + full-catalogue matched L_IC only | — | — | 0.0235 / 0.3249 | 2 |
| sasrec_baseline | SASRec, classic BCE with one negative | — | — | 0.0108 / — | 1 |
| [sasrec_inbatch_l00](configs/train/clothing64/sasrec_inbatch_l00.json) | SASRec, in-batch sampled softmax, no correction | 0.0085 ± 0.0004 / 0.2105 ± 0.0060 | 1,2,3 | 0.0085 / — | 4 |
| [sasrec_inbatch_logq](configs/train/clothing64/sasrec_inbatch_logq.json) | SASRec, in-batch + logQ | 0.0178 ± 0.0012 / 0.3004 ± 0.0035 | 1,2,3 | 0.0155 / — | 4 |

## Beauty — 82 arms

| arm (config) | what it is | seeds: ndcg@20 / recall@1000 | seeds | single run (seed 42): ndcg@20 / recall@1000 | runs |
|---|---|---|---|---|---|
| [01_bs256](configs/train/beauty/01_bs256.json) | 01 with batch size 256 | — | — | 0.0374 / 0.4259 | 1 |
| [01_bs512](configs/train/beauty/01_bs512.json) | 01 with batch size 512 | — | — | 0.0390 / 0.4316 | 1 |
| [01_nousermask](configs/train/beauty/01_nousermask.json) | 01 without the same-user mask | — | — | 0.0382 / 0.4313 | 1 |
| [01_orig](configs/train/beauty/01_orig.json) | MCLSR without graph; in-batch sampled softmax, no correction (λ=0) | 0.0386 ± 0.0014 / 0.4356 ± 0.0027 | 1,2,3 | 0.0380 / 0.4308 | 4 |
| [01_orig_det](configs/train/beauty/01_orig_det.json) | 01 with deterministic kernels (reproducibility check) | 0.0370 ± 0.0000 / 0.4334 ± 0.0000 | 1 | — | 1 |
| [01_orig_det_b](configs/train/beauty/01_orig_det_b.json) | 01 with deterministic kernels, second run | 0.0370 ± 0.0000 / 0.4334 ± 0.0000 | 1 | — | 1 |
| [01_popular127](configs/train/beauty/01_popular127.json) | no graph; 127 popularity-sampled negatives | — | — | 0.0407 / 0.4472 | 1 |
| [01_uniform127](configs/train/beauty/01_uniform127.json) | no graph; 127 uniform negatives per query, no correction | — | — | 0.0491 / 0.4862 | 1 |
| [01_uniform1280](configs/train/beauty/01_uniform1280.json) | no graph; 1280 uniform negatives per query (as in the MCLSR paper) | — | — | 0.0537 / 0.4854 | 1 |
| [02_logq_bs256](configs/train/beauty/02_logq_bs256.json) | 02 with batch size 256 | — | — | 0.0554 / 0.4959 | 1 |
| [02_logq_bs512](configs/train/beauty/02_logq_bs512.json) | 02 with batch size 512 | — | — | 0.0559 / 0.5019 | 1 |
| [02_logq_corrected](configs/train/beauty/02_logq_corrected.json) | 02 with the RecSys'25 corrected form (positive out of the denominator, sg(1−P̂) weight) | 0.0546 ± 0.0018 / 0.5110 ± 0.0042 | 1,2,3 | 0.0554 / 0.5007 | 4 |
| [02_logq_downstream](configs/train/beauty/02_logq_downstream.json) | MCLSR without graph; in-batch + logQ on L_P (λ=1) | 0.0556 ± 0.0006 / 0.5098 ± 0.0061 | 1,2,3 | 0.0555 / 0.5060 | 4 |
| [02_logq_downstream_loo](configs/train/beauty/02_logq_downstream_loo.json) | 02 with leave-own-out q′ | — | — | 0.0546 / 0.4975 | 1 |
| [02_logq_l025](configs/train/beauty/02_logq_l025.json) | 02 with λ=0.25 | — | — | 0.0440 / 0.4581 | 1 |
| [02_logq_l05](configs/train/beauty/02_logq_l05.json) | 02 with λ=0.5 | 0.0513 ± 0.0005 / 0.4867 ± 0.0005 | 1,2,3 | 0.0511 / 0.4767 | 4 |
| [02_logq_nousermask](configs/train/beauty/02_logq_nousermask.json) | 02 without the same-user mask | — | — | 0.0551 / 0.5063 | 1 |
| [02_logq_pos](configs/train/beauty/02_logq_pos.json) | 02 with the standard form (positive corrected too) | 0.0558 ± 0.0014 / 0.5106 ± 0.0061 | 1,2,3 | 0.0551 / 0.5057 | 4 |
| [02_logq_targetq](configs/train/beauty/02_logq_targetq.json) | 02 with q from target counts | — | — | 0.0559 / 0.5086 | 1 |
| [02_mns128](configs/train/beauty/02_mns128.json) | 02 + 128 shared uniform negatives (MNS), mixture proposal | — | — | 0.0572 / 0.5009 | 1 |
| [02_mns128_corr](configs/train/beauty/02_mns128_corr.json) | MNS, corrected form | — | — | 0.0560 / 0.5002 | 1 |
| [02_mns128_std](configs/train/beauty/02_mns128_std.json) | MNS, standard form | — | — | 0.0546 / 0.5071 | 1 |
| [02_popular127_logq](configs/train/beauty/02_popular127_logq.json) | no graph; 127 popularity-sampled negatives + logQ | — | — | 0.0497 / 0.4765 | 1 |
| [03_graph](configs/train/beauty/03_graph.json) | full model: user–item graph + L_IL; logQ on L_P | 0.0567 ± 0.0014 / 0.5399 ± 0.0034 | 1,2,3 | 0.0574 / 0.5484 | 4 |
| [03_graph_a025](configs/train/beauty/03_graph_a025.json) | 03 with interest mix α=0.25 | — | — | 0.0543 / 0.5345 | 1 |
| [03_graph_a075](configs/train/beauty/03_graph_a075.json) | 03 with α=0.75 | — | — | 0.0579 / 0.5379 | 1 |
| [03_graph_a09](configs/train/beauty/03_graph_a09.json) | 03 with α=0.9 | — | — | 0.0585 / 0.5456 | 1 |
| [03_graph_corrected](configs/train/beauty/03_graph_corrected.json) | 03 with the RecSys'25 corrected form on L_P | — | — | 0.0561 / 0.5467 | 1 |
| [03_graph_cosine_t01](configs/train/beauty/03_graph_cosine_t01.json) | 03 with cosine L_IL, τ=0.1 | — | — | 0.0569 / 0.5368 | 1 |
| [03_graph_cosine_t05](configs/train/beauty/03_graph_cosine_t05.json) | 03 with cosine L_IL, τ=0.5 | — | — | 0.0577 / 0.5327 | 1 |
| [03_graph_cosine_t10](configs/train/beauty/03_graph_cosine_t10.json) | 03 with cosine L_IL, τ=1 | — | — | 0.0547 / 0.5308 | 1 |
| [03_graph_depth0](configs/train/beauty/03_graph_depth0.json) | 03 with no graph propagation (raw user / item embeddings feed the general-interest branch) | 0.0575 ± 0.0016 / 0.5146 ± 0.0021 | 1,2,3 | — | 3 |
| [03_graph_depth1](configs/train/beauty/03_graph_depth1.json) | 03 with one propagation layer | 0.0470 ± 0.0000 / 0.4975 ± 0.0000 | 1 | — | 1 |
| [03_graph_euclid_t05](configs/train/beauty/03_graph_euclid_t05.json) | 03 with squared-euclidean L_IL, τ=0.5 | — | — | 0.0589 / 0.5398 | 1 |
| [03_graph_euclid_t1](configs/train/beauty/03_graph_euclid_t1.json) | 03 with squared-euclidean L_IL, τ=1 | — | — | 0.0574 / 0.5405 | 1 |
| [03_graph_euclid_t20](configs/train/beauty/03_graph_euclid_t20.json) | 03 with squared-euclidean L_IL, τ=2 | — | — | 0.0599 / 0.5367 | 1 |
| [03_graph_gd0](configs/train/beauty/03_graph_gd0.json) | 03 without graph edge dropout | 0.0546 ± 0.0000 / 0.5323 ± 0.0000 | 1 | — | 1 |
| [03_graph_il00](configs/train/beauty/03_graph_il00.json) | graph without L_IL (β=0) | 0.0500 ± 0.0007 / 0.4977 ± 0.0022 | 1,2,3 | 0.0537 / 0.4924 | 4 |
| [03_graph_il025](configs/train/beauty/03_graph_il025.json) | 03 with β=0.25 | — | — | 0.0569 / 0.5454 | 1 |
| [03_graph_il05](configs/train/beauty/03_graph_il05.json) | 03 with L_IL weight β=0.5 | 0.0580 ± 0.0004 / 0.5405 ± 0.0077 | 1,2,3 | 0.0591 / 0.5317 | 4 |
| [03_graph_il20](configs/train/beauty/03_graph_il20.json) | 03 with β=2 | — | — | 0.0532 / 0.5151 | 1 |
| [03_graph_l00](configs/train/beauty/03_graph_l00.json) | graph + L_IL without the correction (λ=0) — factorial cell | 0.0393 ± 0.0010 / 0.4735 ± 0.0069 | 1,2,3 | — | 3 |
| [03_graph_mean](configs/train/beauty/03_graph_mean.json) | 03 with the LightGCN layer mean (layers 0..2 averaged) instead of the last layer | 0.0552 ± 0.0000 / 0.5240 ± 0.0000 | 1 | — | 1 |
| [03_graph_mean_gd0](configs/train/beauty/03_graph_mean_gd0.json) | 03 with the layer mean and no edge dropout | 0.0532 ± 0.0000 / 0.5099 ± 0.0000 | 1 | — | 1 |
| [03_graph_paper_faithful](configs/train/beauty/03_graph_paper_faithful.json) | 03 with shared projector, cosine L_IL and the paper scheme | — | — | 0.0569 / 0.5154 | 1 |
| [03_graph_sharedproj](configs/train/beauty/03_graph_sharedproj.json) | 03 with a shared L_IL projector | — | — | 0.0565 / 0.5357 | 1 |
| [04_graph_logq_lil](configs/train/beauty/04_graph_logq_lil.json) | as 03, logQ also on L_IL | 0.0508 ± 0.0005 / 0.5294 ± 0.0029 | 1,2,3 | 0.0522 / 0.5293 | 4 |
| [04_graph_logq_lil_centered](configs/train/beauty/04_graph_logq_lil_centered.json) | 04 with centred log q on L_IL | — | — | 0.0564 / 0.5446 | 1 |
| [04_graph_logq_lil_cosine_t05](configs/train/beauty/04_graph_logq_lil_cosine_t05.json) | 04 with cosine L_IL, τ=0.5 | — | — | 0.0566 / 0.5284 | 1 |
| [04_graph_logq_lil_euclid_t05](configs/train/beauty/04_graph_logq_lil_euclid_t05.json) | 04 with squared-euclidean L_IL, τ=0.5 | — | — | 0.0563 / 0.5322 | 1 |
| [04_graph_logq_lil_euclid_t10](configs/train/beauty/04_graph_logq_lil_euclid_t10.json) | 04 with squared-euclidean L_IL, τ=1 | — | — | 0.0550 / 0.5257 | 1 |
| [04_graph_logq_lil_euclid_t20](configs/train/beauty/04_graph_logq_lil_euclid_t20.json) | 04 with squared-euclidean L_IL, τ=2 | — | — | 0.0540 / 0.5344 | 1 |
| [04_graph_logq_lil_l01](configs/train/beauty/04_graph_logq_lil_l01.json) | 04 with λ_IL=0.1 | — | — | 0.0565 / 0.5371 | 1 |
| [04_graph_logq_lil_l03](configs/train/beauty/04_graph_logq_lil_l03.json) | 04 with λ_IL=0.3 | — | — | 0.0544 / 0.5331 | 1 |
| [04_graph_logq_lil_uniformq](configs/train/beauty/04_graph_logq_lil_uniformq.json) | 04 with a constant user-count table (pure margin, no popularity information) | 0.0524 ± 0.0014 / 0.5276 ± 0.0026 | 1,2,3 | 0.0521 / 0.5206 | 4 |
| [05_full_baseline](configs/train/beauty/05_full_baseline.json) | paper model: graph + L_IL + L_UC + L_IC (γ=0.05); logQ on L_P | 0.0573 ± 0.0012 / 0.5390 ± 0.0033 | 1,2,3 | 0.0563 / 0.5344 | 4 |
| [05_full_l00](configs/train/beauty/05_full_l00.json) | paper model (graph + L_IL + L_UC + L_IC) without the correction (λ=0) | 0.0374 ± 0.0005 / 0.4724 ± 0.0081 | 1,2,3 | — | 3 |
| [05_full_mean](configs/train/beauty/05_full_mean.json) | paper model with the LightGCN layer mean instead of the last layer | 0.0561 ± 0.0000 / 0.5223 ± 0.0000 | 1 | — | 1 |
| [05_full_uw](configs/train/beauty/05_full_uw.json) | paper model with learned (uncertainty) loss weights | — | — | 0.0437 / 0.5029 | 1 |
| [05_full_w01](configs/train/beauty/05_full_w01.json) | paper model with γ=0.1 | — | — | 0.0565 / 0.5485 | 1 |
| [05_full_w02](configs/train/beauty/05_full_w02.json) | paper model with γ=0.2 | — | — | 0.0557 / 0.5348 | 1 |
| [05_full_w05](configs/train/beauty/05_full_w05.json) | paper model with γ=0.5 | 0.0563 ± 0.0010 / 0.5350 ± 0.0005 | 1,2,3 | 0.0585 / 0.5353 | 4 |
| [05_minus_ic](configs/train/beauty/05_minus_ic.json) | paper model minus the item–item graph loss L_IC | 0.0570 ± 0.0004 / 0.5407 ± 0.0029 | 1,2,3 | — | 3 |
| [05_minus_il](configs/train/beauty/05_minus_il.json) | paper model minus the alignment loss L_IL | 0.0516 ± 0.0017 / 0.4896 ± 0.0036 | 1,2,3 | — | 3 |
| [05_minus_uc](configs/train/beauty/05_minus_uc.json) | paper model minus the user–user graph loss L_UC | 0.0565 ± 0.0003 / 0.5347 ± 0.0068 | 1,2,3 | — | 3 |
| [06_full_logq_ucic](configs/train/beauty/06_full_logq_ucic.json) | paper model, logQ also on L_UC / L_IC | 0.0557 ± 0.0019 / 0.5398 ± 0.0041 | 1,2,3 | 0.0537 / 0.5389 | 4 |
| [07_full_logq_all](configs/train/beauty/07_full_logq_all.json) | paper model, logQ on every loss | — | — | 0.0516 / 0.5246 | 1 |
| [09_item_only](configs/train/beauty/09_item_only.json) | graph + L_IC only (no L_IL, no L_UC) | — | — | 0.0535 / 0.4876 | 1 |
| [10_item_only_logq](configs/train/beauty/10_item_only_logq.json) | graph + L_IC with logQ | — | — | 0.0548 / 0.5024 | 1 |
| [10_item_only_logq_ctxq_v2](configs/train/beauty/10_item_only_logq_ctxq_v2.json) | graph + L_IC with logQ from context-inclusion counts | — | — | 0.0563 / 0.5045 | 1 |
| [14_full_softmax](configs/train/beauty/14_full_softmax.json) | exact full-catalogue softmax on L_P (no sampling), no graph | 0.0577 ± 0.0013 / 0.5100 ± 0.0042 | 1,2,3 | 0.0571 / 0.5062 | 4 |
| [14_full_softmax_graph](configs/train/beauty/14_full_softmax_graph.json) | exact full-catalogue softmax on L_P, with graph + L_IL | — | — | 0.0576 / 0.5465 | 1 |
| [17_user_only_full_matched](configs/train/beauty/17_user_only_full_matched.json) | graph + full-catalogue matched L_UC only | — | — | 0.0525 / 0.4904 | 1 |
| [18_item_only_full_matched](configs/train/beauty/18_item_only_full_matched.json) | graph + full-catalogue matched L_IC only | — | — | 0.0516 / 0.4939 | 1 |
| [sasrec_baseline](configs/train/beauty/sasrec_baseline.json) | SASRec, classic BCE with one negative | — | — | 0.0393 / 0.4579 | 1 |
| [sasrec_inbatch_corrected](configs/train/beauty/sasrec_inbatch_corrected.json) | SASRec, RecSys'25 corrected form | 0.0632 ± 0.0007 / 0.4999 ± 0.0010 | 1,2,3 | 0.0656 / 0.5034 | 4 |
| [sasrec_inbatch_l00](configs/train/beauty/sasrec_inbatch_l00.json) | SASRec, in-batch sampled softmax, no correction | 0.0478 ± 0.0030 / 0.4247 ± 0.0045 | 1,2,3 | 0.0446 / 0.4178 | 4 |
| [sasrec_inbatch_l00_umask](configs/train/beauty/sasrec_inbatch_l00_umask.json) | SASRec λ=0 with the same-user mask | — | — | 0.0475 / 0.4178 | 1 |
| [sasrec_inbatch_logq](configs/train/beauty/sasrec_inbatch_logq.json) | SASRec, in-batch + logQ | 0.0631 ± 0.0012 / 0.5002 ± 0.0031 | 1,2,3 | 0.0667 / 0.5067 | 4 |
| [sasrec_inbatch_logq_umask](configs/train/beauty/sasrec_inbatch_logq_umask.json) | SASRec λ=1 with the same-user mask | — | — | 0.0636 / 0.4996 | 1 |
| [sasrec_ladder_l00](configs/train/beauty/sasrec_ladder_l00.json) | SASRec λ=0 on the MCLSR prefix ladder (last-position query) | — | — | 0.0408 / 0.4017 | 1 |
| [sasrec_ladder_logq](configs/train/beauty/sasrec_ladder_logq.json) | SASRec λ=1 on the prefix ladder | — | — | 0.0605 / 0.4675 | 1 |

## CDs & Vinyl ("Toys") — 29 arms

| arm (config) | what it is | seeds: ndcg@20 / recall@1000 | seeds | single run (seed 42): ndcg@20 / recall@1000 | runs |
|---|---|---|---|---|---|
| [01_orig](configs/train/toys/01_orig.json) | MCLSR without graph; in-batch sampled softmax, no correction (λ=0) | 0.0349 ± 0.0007 / 0.4126 ± 0.0033 | 1,2,3 | 0.0346 / 0.4090 | 5 |
| [02_logq_corrected](configs/train/toys/02_logq_corrected.json) | 02 with the RecSys'25 corrected form (positive out of the denominator, sg(1−P̂) weight) | — | — | 0.0484 / 0.4818 | 1 |
| [02_logq_downstream](configs/train/toys/02_logq_downstream.json) | MCLSR without graph; in-batch + logQ on L_P (λ=1) | 0.0497 ± 0.0009 / 0.4806 ± 0.0015 | 1,2,3 | 0.0492 / 0.4767 | 5 |
| [03_graph](configs/train/toys/03_graph.json) | full model: user–item graph + L_IL; logQ on L_P | 0.0497 ± 0.0005 / 0.5075 ± 0.0030 | 1,2,3 | 0.0499 / 0.5040 | 4 |
| [03_graph_depth0](configs/train/toys/03_graph_depth0.json) | 03 with no graph propagation (raw user / item embeddings feed the general-interest branch) | 0.0509 ± 0.0000 / 0.4999 ± 0.0000 | 1 | — | 1 |
| [03_graph_euclid_t1](configs/train/toys/03_graph_euclid_t1.json) | 03 with squared-euclidean L_IL, τ=1 | — | — | 0.0485 / 0.4986 | 1 |
| [03_graph_il00](configs/train/toys/03_graph_il00.json) | graph without L_IL (β=0) | 0.0390 ± 0.0009 / 0.4454 ± 0.0017 | 1,2,3 | — | 3 |
| [03_graph_l00](configs/train/toys/03_graph_l00.json) | graph + L_IL without the correction (λ=0) — factorial cell | 0.0376 ± 0.0004 / 0.4431 ± 0.0023 | 1,2,3 | — | 3 |
| [04_graph_logq_lil](configs/train/toys/04_graph_logq_lil.json) | as 03, logQ also on L_IL | 0.0489 ± 0.0007 / 0.5054 ± 0.0035 | 1,2,3 | 0.0485 / 0.5031 | 4 |
| [04_graph_logq_lil_uniformq](configs/train/toys/04_graph_logq_lil_uniformq.json) | 04 with a constant user-count table (pure margin, no popularity information) | 0.0489 ± 0.0003 / 0.5052 ± 0.0005 | 1,2,3 | — | 3 |
| [05_full_baseline](configs/train/toys/05_full_baseline.json) | paper model: graph + L_IL + L_UC + L_IC (γ=0.05); logQ on L_P | 0.0494 ± 0.0004 / 0.5064 ± 0.0037 | 1,2,3 | 0.0488 / 0.5059 | 4 |
| [05_full_l00](configs/train/toys/05_full_l00.json) | paper model (graph + L_IL + L_UC + L_IC) without the correction (λ=0) | 0.0385 ± 0.0003 / 0.4474 ± 0.0015 | 1,2,3 | — | 3 |
| [05_minus_ic](configs/train/toys/05_minus_ic.json) | paper model minus the item–item graph loss L_IC | 0.0488 ± 0.0007 / 0.5052 ± 0.0027 | 1,2,3 | — | 3 |
| [05_minus_il](configs/train/toys/05_minus_il.json) | paper model minus the alignment loss L_IL | 0.0404 ± 0.0008 / 0.4487 ± 0.0017 | 1,2,3 | — | 3 |
| [05_minus_uc](configs/train/toys/05_minus_uc.json) | paper model minus the user–user graph loss L_UC | 0.0494 ± 0.0005 / 0.5053 ± 0.0020 | 1,2,3 | — | 3 |
| [06_full_logq_ucic](configs/train/toys/06_full_logq_ucic.json) | paper model, logQ also on L_UC / L_IC | 0.0498 ± 0.0007 / 0.5054 ± 0.0025 | 1,2,3 | 0.0478 / 0.5014 | 4 |
| [09_item_only](configs/train/toys/09_item_only.json) | graph + L_IC only (no L_IL, no L_UC) | — | — | 0.0407 / 0.4533 | 1 |
| [10_item_only_logq](configs/train/toys/10_item_only_logq.json) | graph + L_IC with logQ | — | — | 0.0390 / 0.4308 | 1 |
| [14_full_softmax](configs/train/toys/14_full_softmax.json) | exact full-catalogue softmax on L_P (no sampling), no graph | 0.0534 ± 0.0007 / 0.4939 ± 0.0006 | 1,2,3 | — | 3 |
| [17_user_only_full_matched](configs/train/toys/17_user_only_full_matched.json) | graph + full-catalogue matched L_UC only | — | — | 0.0365 / 0.4313 | 1 |
| [18_item_only_full_matched](configs/train/toys/18_item_only_full_matched.json) | graph + full-catalogue matched L_IC only | — | — | 0.0401 / 0.4437 | 1 |
| [sasrec_baseline](configs/train/toys/sasrec_baseline.json) | SASRec, classic BCE with one negative | — | — | 0.0239 / 0.3850 | 1 |
| [sasrec_inbatch_corrected](configs/train/toys/sasrec_inbatch_corrected.json) | SASRec, RecSys'25 corrected form | — | — | 0.0429 / — | 1 |
| [sasrec_inbatch_l00](configs/train/toys/sasrec_inbatch_l00.json) | SASRec, in-batch sampled softmax, no correction | 0.0268 ± 0.0006 / 0.3784 ± 0.0055 | 1,2,3 | 0.0240 / — | 4 |
| [sasrec_inbatch_l00_umask](configs/train/toys/sasrec_inbatch_l00_umask.json) | SASRec λ=0 with the same-user mask | — | — | 0.0286 / — | 1 |
| [sasrec_inbatch_logq](configs/train/toys/sasrec_inbatch_logq.json) | SASRec, in-batch + logQ | 0.0438 ± 0.0004 / 0.4642 ± 0.0044 | 1,2,3 | 0.0428 / — | 4 |
| [sasrec_inbatch_logq_umask](configs/train/toys/sasrec_inbatch_logq_umask.json) | SASRec λ=1 with the same-user mask | — | — | 0.0430 / — | 1 |
| [sasrec_ladder_l00](configs/train/toys/sasrec_ladder_l00.json) | SASRec λ=0 on the MCLSR prefix ladder (last-position query) | — | — | 0.0283 / — | 1 |
| [sasrec_ladder_logq](configs/train/toys/sasrec_ladder_logq.json) | SASRec λ=1 on the prefix ladder | — | — | 0.0405 / — | 1 |

Total training runs indexed: 304. Re-run summaries: `python scripts/summarize_runs.py --pattern "<prefix>_*" --report eval/ndcg@20 eval/recall@1000` over `tensorboard_logs/`, and `python scripts/summarize_seeds.py --prefix <beauty|toys|clothing64>`.
