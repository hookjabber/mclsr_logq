# Paired per-user comparisons — beauty

Validation-selected checkpoints (ndcg@20) of both arms scored on the same test users; differences pooled over seeds; CI by user bootstrap (10 000 draws, all seeds of a user resampled together); p by sign-flip permutation.

| arm − baseline | metric | per-seed Δ | mean Δ | 95% CI (user bootstrap) | p (sign-flip) | users × seeds |
|---|---|---|---|---|---|---|
| 02_logq_downstream − 01_orig | ndcg@20 | +0.0182, +0.0155, +0.0171 | +0.0169 | [+0.0136, +0.0203] | 0.0001 | 2237 × 3 |
| 02_logq_downstream − 01_orig | recall@1000 | +0.0758, +0.0775, +0.0717 | +0.0750 | [+0.0636, +0.0865] | 0.0001 | 2237 × 3 |
| 03_graph − 03_graph_l00 | ndcg@20 | +0.0153, +0.0183, +0.0186 | +0.0174 | [+0.0142, +0.0208] | 0.0001 | 2237 × 3 |
| 03_graph − 03_graph_l00 | recall@1000 | +0.0724, +0.0781, +0.0706 | +0.0737 | [+0.0637, +0.0837] | 0.0001 | 2237 × 3 |
| 05_full_baseline − 05_full_l00 | ndcg@20 | +0.0189, +0.0216, +0.0192 | +0.0199 | [+0.0168, +0.0231] | 0.0001 | 2237 × 3 |
| 05_full_baseline − 05_full_l00 | recall@1000 | +0.0831, +0.0721, +0.0737 | +0.0763 | [+0.0659, +0.0866] | 0.0001 | 2237 × 3 |
| 04_graph_logq_lil − 03_graph | ndcg@20 | -0.0045, -0.0063, -0.0070 | -0.0059 | [-0.0079, -0.0040] | 0.0001 | 2237 × 3 |
| 04_graph_logq_lil − 03_graph | recall@1000 | -0.0092, -0.0107, -0.0099 | -0.0099 | [-0.0164, -0.0033] | 0.0038 | 2237 × 3 |
| 14_full_softmax − 02_logq_downstream | ndcg@20 | +0.0028, +0.0010, +0.0026 | +0.0021 | [-0.0007, +0.0049] | 0.1338 | 2237 × 3 |
| 14_full_softmax − 02_logq_downstream | recall@1000 | +0.0032, +0.0035, -0.0028 | +0.0013 | [-0.0068, +0.0094] | 0.7474 | 2237 × 3 |
| 05_full_baseline − 03_graph | ndcg@20 | +0.0014, +0.0023, -0.0018 | +0.0006 | [-0.0007, +0.0019] | 0.3629 | 2237 × 3 |
| 05_full_baseline − 03_graph | recall@1000 | +0.0008, +0.0010, -0.0020 | -0.0001 | [-0.0055, +0.0052] | 0.9727 | 2237 × 3 |
| sasrec_inbatch_logq − sasrec_inbatch_l00 | ndcg@20 | +0.0199, +0.0118, +0.0140 | +0.0152 | [+0.0118, +0.0188] | 0.0001 | 2237 × 3 |
| sasrec_inbatch_logq − sasrec_inbatch_l00 | recall@1000 | +0.1076, +0.0882, +0.0829 | +0.0929 | [+0.0816, +0.1042] | 0.0001 | 2237 × 3 |
