# Configs

One JSON per experiment arm; the file is the complete specification of a run (data, graphs, model,
losses, optimiser, validation protocol). `tests/test_configs.py` validates every maintained config
against the component registry.

## Directories

| directory | what |
|---|---|
| `train/beauty/`, `train/toys/` (CDs & Vinyl), `train/clothing64/` | the arms of the multi-seed study, validation every 64 steps (256 on CDs & Vinyl) |
| `train/grid/` | the original single-seed Clothing grid (validation every 256 steps) |
| `train/legacy/` | historical configs, kept for provenance, not validated |

## Arm numbering

| prefix | question |
|---|---|
| `01_*` | MCLSR without the graph branch, in-batch sampled softmax without correction (λ = 0) |
| `02_*` | the same with the logQ correction on the retrieval loss L_P (λ = 1); variants: correction form, q table, batch size, mixed negatives |
| `03_*` | user–item graph + interest-level contrastive L_IL, logQ on L_P; variants: depth, layer mean, edge dropout, cosine / euclidean / paper-form L_IL, β |
| `04_*` | 03 with the correction also on L_IL (the harmful placement); variants: dose λ, constant q, centred, cosine |
| `05_*` | the full model of the paper (all three graphs, all four losses); `_l00` = no correction, `_paper` = the paper's cosine form of the contrastive losses, `_w*` = weight sweeps, `_minus_*` = component ablation |
| `06_*`, `07_*` | the correction on the feature-level losses L_UC / L_IC, and on all losses |
| `09_*`, `10_*` | item-level contrastive only, with / without correction |
| `14_*` | the exact full-catalogue softmax on L_P (the no-sampling anchor) |
| `17_*`, `18_*` | matched full-catalogue contrastive anchors for the user / item views |
| `sasrec_*` | SASRec with the same retrieval loss: `_inbatch_l00` / `_inbatch_logq`, `_corrected` (RecSys'25 form), `_umask` (same-user mask), `_ladder` (the MCLSR training protocol), `_baseline` (classic BCE) |

## Knobs

Dataset (`dataset`): `type` (`mclsr` sequences, `graph` wrapper around them, `sasrec_comparison`),
`max_sequence_length`, `use_user_graph`, `use_item_graph`, `neighborhood_size` (K = 50),
`use_train_data_only` (must stay true), `eval_history_max_len`.

Model (`model`, type `mclsr`): `use_graph`, `num_graph_layers`, `graph_dropout`, `graph_layer_mean`
(LightGCN layer mean instead of the last layer), `share_il_projector` (one projector for both L_IL
views, paper eq. 7), `alpha` (eq. 13), `dropout`, `layer_norm_eps`, `initializer_range`,
`eval_top_k`. Type `sasrec_in_batch`: `num_layers`, `num_heads`, `dim_feedforward`, `dropout`,
`query_positions` (`all` = standard SASRec, `last` = the ladder protocol).

Losses (`loss.losses[]`, combined by `composite` with `weight` per term):

| type | role | keys |
|---|---|---|
| `mclsr_logq_inbatch` | retrieval loss L_P with in-batch negatives | `logq_lambda` (correction strength), `variant` (`negatives_only` / `standard` / `corrected`), `leave_own_out`, `mixed_uniform_negatives` + `table_prefix`, `normalize_embeddings`, `temperature`, `path_to_item_counts`, `user_ids_prefix` (same-user mask when set) |
| `full_softmax` | exact full-catalogue softmax anchor | `queries_prefix`, `table_prefix`, `positive_ids_prefix` |
| `sampled_softmax`, `mclsr_logq_special` | explicit sampled negatives (uniform / popular), with or without correction | `negative_prefix`, `path_to_item_counts`, `logq_lambda` |
| `fps` | two-view contrastive (L_IL, L_UC, L_IC) | `temperature` (τ), `scheme` (`symmetric` default / `paper` = eq. 8 and 11 / `cross_only`), `normalize_embeddings` (cosine), `similarity` (`dot` / `euclidean`) |
| `fps_logq` | the correction on a two-view contrastive loss | `fps` keys + `logq_lambda`, `path_to_counts`, `ids_prefix`, `logq_probability_mode`, `leave_own_out`, `center_log_q` |
| `matched_contrastive_full_softmax`, `contrastive_full_softmax` | full-catalogue contrastive anchors | `*_table_prefix`, `ids_prefix`, `temperature`, `path_to_train_presence` |

Protocol (`callback`, `best_metric`, `epochs_threshold`, `train_steps_num`): validation every
`on_step` steps, best checkpoint by `validation/ndcg@20` (and `validation/recall@1000` in the
confirmatory runner), early stop after `epochs_threshold` epochs without improvement, a hard step cap.
