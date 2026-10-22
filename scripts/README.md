# Scripts

Every script prints its usage in its docstring (`python scripts/<name>.py --help` or the top of the
file). Run them from the repository root with the framework on the path: `PYTHONPATH=src`.

## Data preparation

| script | what |
|---|---|
| `prepare_dataset.py` | raw ratings csv (or a 5-core sequence file) → the split files of the study; `--verify-against` diffs the result with existing files |
| `generate_item_counts.py` | item frequency table for the logQ correction (`item_counts.pkl`) |
| `generate_user_counts.py` | user frequency table (user-level correction experiments) |
| `generate_mclsr_role_counts.py` | role-exact tables: how often an item is a target / a context item of the ladder |
| `generate_train_presence.py` | train-presence masks of items / users (matched full-catalogue contrastive arms) |
| `check_split_leakage.py` | the user split is disjoint, the count tables match the files, the graphs contain only training pairs |
| `gen_beauty_configs.py` | generator of the Beauty arm configs from the Clothing templates |

## Training

| script | what |
|---|---|
| `train_confirmatory.py` | pre-registered runner: test callback stripped, one test read per validation-selected checkpoint, JSON report with sha256 of config, tables and checkpoints |
| `run_queue_seeds.sh` | several arms × several seeds through the confirmatory runner, with done-markers and logs |
| `run_queue_generic.sh` | exploratory runs (`python -m irec.train`), same markers and logs |

## Evaluation and summaries

| script | what |
|---|---|
| `eval_checkpoint.py` | score any saved checkpoint on validation or test with the config's metrics |
| `summarize_seeds.py` | mean ± std over seeds from the confirmatory reports, plus paired per-seed differences |
| `summarize_runs.py` | the exploratory runs from TensorBoard logs under the validation-best protocol |
| `paired_bootstrap.py` | per-user paired bootstrap intervals and sign-flip p-values for arm − baseline pairs |
| `val_test_curves.py` | validation and test ndcg@20 along training for runs matching a pattern |
| `decile_recall.py` | popularity-stratified recall from a checkpoint |
| `popularity_baseline.py` | MostPop anchor on the test protocol, with an independent numpy recomputation of the metrics |

## Verification (no framework code reused on the reference side)

| script | what |
|---|---|
| `verify_graphs.py` | the cached graphs vs scipy recomputation from the ladder: eq. 1, co-action weights, top-50 |
| `indep_check_architecture.py` | the MCLSR forward pass, every loss and the eval top-k in numpy / float64 on a real batch |
| `indep_check_sasrec.py` | the SASRec branch: numpy transformer, identical eval inputs, causality, padding, loss |
| `indep_check_beauty.py`, `indep_check_clothing.py` | the headline effect reproduced by a standalone retrieval model without the framework |
| `diagnose_feature_graphs.py` | graph-neighborhood overlap, checkpoint-view similarity and logged loss trajectories; not a test of recommendation benefit |

## Reporting

| script | what |
|---|---|
| `plot_study_figures.py` | the figures of RESULTS.md Part A from the confirmatory reports |
| `plot_paired_bootstrap.py` | the interval plot of the paired comparisons |
| `build_run_index.py` | RUN_INDEX.md: every run of the study with its numbers and config link |
