# Paired per-user comparisons — beauty

Validation-selected checkpoints (ndcg@20) of both arms scored on the same test users; differences pooled over seeds; CI by user bootstrap (10 000 draws, all seeds of a user resampled together); p by sign-flip permutation.

| arm − baseline | metric | per-seed Δ | mean Δ | 95% CI (user bootstrap) | p (sign-flip) | users × seeds |
|---|---|---|---|---|---|---|
| 05_full_paper − 05_full_l00_paper | ndcg@20 | +0.0159, +0.0116, +0.0138 | +0.0138 | [+0.0106, +0.0170] | 0.0001 | 2237 × 3 |
| 05_full_paper − 05_full_l00_paper | recall@1000 | +0.0566, +0.0506, +0.0578 | +0.0550 | [+0.0455, +0.0649] | 0.0001 | 2237 × 3 |
| 05_full_l00_paper − 05_full_l00 | ndcg@20 | +0.0025, +0.0064, +0.0061 | +0.0050 | [+0.0023, +0.0078] | 0.0003 | 2237 × 3 |
| 05_full_l00_paper − 05_full_l00 | recall@1000 | +0.0199, +0.0146, +0.0106 | +0.0150 | [+0.0059, +0.0243] | 0.0025 | 2237 × 3 |
| 05_full_paper − 05_full_baseline | ndcg@20 | -0.0005, -0.0036, +0.0007 | -0.0011 | [-0.0036, +0.0013] | 0.3617 | 2237 × 3 |
| 05_full_paper − 05_full_baseline | recall@1000 | -0.0066, -0.0069, -0.0053 | -0.0062 | [-0.0136, +0.0012] | 0.0975 | 2237 × 3 |
