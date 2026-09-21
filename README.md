
# IRec × MCLSR: a systematic study of the logQ sampling-bias correction
<p align="center">
   <a>
      <img src="./assets/irec_logo.png" alt="cool irec logo" width="40%" height="40%">
   </a>
</p>

<p align="center">
   <a href="../../actions/workflows/tests.yml">
      <img src="../../actions/workflows/tests.yml/badge.svg" alt="Tests">
   </a>
   <a>
      <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python version">
   </a>
   <a href="https://github.com/astral-sh/ruff">
      <img src="https://img.shields.io/badge/code_style-ruff-blue?logo=python" alt="Code style">
   </a>
   <a>
      <img src="https://img.shields.io/badge/license-Apache 2.0-blue?logo=apache" alt="License">
   </a>
</p>

**IRec** is a config-driven PyTorch framework for reproducible research in sequential and graph-based recommendation.

This repository hosts a systematic study of the **logQ sampling-bias correction** in a
multi-loss graph-contrastive recommender (MCLSR, CIKM'22): where the correction helps
(in-batch retrieval), where it does not (contrastive alignment losses), and why. Every
loss variant is locked in by reference tests, and every experiment is a single JSON
config away. **All experiment results with verdicts: [RESULTS.md](./RESULTS.md).**

## Start here: the study at a glance

![logQ correction on the retrieval loss: three datasets, two encoders, mean ± std over 3 seeds](./assets/headline_three_datasets.png)

Test ndcg@20 / recall@1000 over the full catalogue, mean of three seeds (model selected on validation, test read once):

| | Clothing | Beauty | CDs & Vinyl ("Toys" in the MCLSR paper) |
|---|---|---|---|
| MCLSR, in-batch sampled softmax, no correction | 0.0152 / 0.252 | 0.0386 / 0.436 | 0.0349 / 0.413 |
| **MCLSR, in-batch + logQ on the retrieval loss** | **0.0226 / 0.314** | **0.0556 / 0.510** | **0.0497 / 0.481** |
| MCLSR, exact full-catalogue softmax | 0.0219 / 0.311 | 0.0577 / 0.510 | 0.0534 / 0.494 |
| full MCLSR (graph + alignment loss), logQ on retrieval | 0.0261 / 0.358 | 0.0567 / 0.540 | 0.0497 / 0.508 |
| full MCLSR, logQ also on the alignment loss | 0.0249 / 0.344 | 0.0508 / 0.529 | 0.0489 / 0.505 |
| SASRec, in-batch: no correction → logQ | 0.0085 → 0.0178 | 0.0478 → 0.0631 | 0.0268 → 0.0438 |

1. The correction on the retrieval loss gives +42…+48 % ndcg@20 on MCLSR and +32…+108 % on SASRec, positive on every seed of every dataset, and recovers 80–100 % of the exact-softmax gain.
2. The same correction on the contrastive alignment loss hurts; the mechanism is a margin, not a sampling correction (a constant-q "correction" reproduces the harm).
3. The graph and the correction are additive; the graph alone does not repair the popularity bias and adds tail recall only. The user–user and item–item graphs of the paper contribute nothing; the alignment loss is what makes the graph useful.
4. The three published forms of the correction (negatives-only, Yi et al., Khrylchenko–Baikalov et al. RecSys'25) are indistinguishable here; the loss weights of the original paper are optimal within noise.

Where to look:

- [RESULTS.md](./RESULTS.md) — figures, headline tables, mechanism and sensitivity tables, component ablation (Part A); the original single-seed Clothing study (Part B).
- [RUN_INDEX.md](./RUN_INDEX.md) — every training run of the study (≈150 arms, ≈280 runs), labelled, with links to its config.
- Per-run reports with content hashes: [results/confirm/](./results/confirm/).
- Code: the correction variants and the exact softmax in [src/irec/loss/logq.py](./src/irec/loss/logq.py) (`MCLSRLogqInBatchLoss` with the three forms, `FpsLogQLoss`, `FullSoftmaxLoss`), the model in [src/irec/models/mclsr.py](./src/irec/models/mclsr.py), the confirmatory runner [scripts/train_confirmatory.py](./scripts/train_confirmatory.py), and a hundred-line independent replication of the headline effect with no framework code, [scripts/indep_check_beauty.py](./scripts/indep_check_beauty.py).

> **Attribution.** IRec is a shared research framework developed by our team (started at the ITMO CT Machine Learning Lab); this repository is my working copy. The from-scratch MCLSR reimplementation and the logQ study here are my own contribution.

## Repository layout

| path | what |
|---|---|
| `configs/train/beauty/`, `configs/train/toys/`, `configs/train/clothing64/` | the experiment arms of the multi-seed study (numbered, one question per config; `toys` = CDs & Vinyl) |
| `configs/train/grid/` | the original Clothing grid (single-seed study, validation every 256 steps) |
| `configs/train/legacy/` | historical configs kept for provenance |
| `results/confirm/`, `results/deciles/`, `results/explor/` | confirmatory seed reports (json with sha256), decile analyses, TensorBoard summaries of the exploratory runs |
| `RUN_INDEX.md` | every training run of the study, labelled, with its numbers and a link to its config |
| `src/irec/` | framework: models, losses, datasets, metrics, callbacks |
| `scripts/` | count-table generation, confirmatory runner, run and seed summaries, checkpoint evaluation, data checks, figures |
| `tests/` | reference tests for the logQ losses + config validation (run in CI) |
| `notebooks/` | dataset preprocessing |

## Installation

### Using uv (Recommended)

1. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.12
   source ./.venv/bin/activate
   ```

2. Install dependencies:

   **For development**
   ```bash
   uv sync --all-extras --frozen
   ```

   **For production**
   ```bash
   uv sync --frozen
   ```

## Preparing datasets
The data splits are generated from the public Amazon review dumps by the Jupyter
notebooks in [notebooks](./notebooks) — run the dataset notebook (e.g.
`AmazonClothingDatasetUserSplit.ipynb`) to produce the `.txt` splits under
`data/`, then generate the count tables and masks as shown in the
reproduce section below.

## Model training
To train a model, simply run the following from the root directory:
```shell
train --params /path/to/config
```

The script has 1 input argument: `params` which is the path to the json file with model configuration. The example of such file can be found [here](./configs). This directory contains json files with model hyperparameters and data preparation instructions. It should contain the following keys:

-`experiment_name` Name of the experiment

-`dataset` Information about the dataset

-`dataloader` Settings for dataloader

-`model` Model hyperparameters

-`optimizer` Optimizer hyperparameters

-`loss` Naming of different loss components

-`callbacks` Different additional traning 

-`use_wandb` Enable Weights & Biases logging for experiment tracking

## Tests

Every logQ loss variant (q / q' / λ=0, both masking modes, cross-only scheme, cosine
scoring, the full-softmax anchors) is checked against an independent naive reference
implementation — values, plus gradient sanity (finiteness and masked-entry
zero-grad checks). Config validation catches a broken config in
seconds instead of hours into a run. Both suites run in CI on every push:

```bash
python tests/test_logq_losses.py
python tests/test_configs.py
```

## Reproducing the logQ study

```bash
# 1. Environment (Python >= 3.12)
uv venv && uv pip install -e .

# 2. Data: run notebooks/AmazonClothingDatasetUserSplit.ipynb -> data/Clothing/*.txt

# 3. Count tables for the logQ correction
python scripts/generate_item_counts.py --input data/Clothing/train_sasrec.txt \
    --output data/Clothing/item_counts.pkl --num_items 23033
python scripts/generate_user_counts.py --input data/Clothing/train_mclsr.txt \
    --output data/Clothing/user_counts.pkl

# 3b. Train-presence masks (matched full-catalog configs 17/18) and role-exact
#     tables (02_logq_targetq / 10_*_ctxq*)
python scripts/generate_train_presence.py --mode item --input data/Clothing/train_sasrec.txt \
    --output data/Clothing/train_presence_items.pkl --num_entities 23033
python scripts/generate_train_presence.py --mode user --input data/Clothing/train_mclsr.txt \
    --output data/Clothing/train_presence_users.pkl --num_entities 39387
python scripts/generate_mclsr_role_counts.py --input data/Clothing/train_mclsr.txt \
    --num_items 23033 --max_len 20 \
    --target_output data/Clothing/item_target_counts.pkl \
    --context_output data/Clothing/item_context_counts.pkl

# 4. Loss correctness tests (reference implementations) and lint
python -m pytest -q tests/        # or: python tests/test_logq_losses.py etc., as in CI
ruff check src scripts tests

# 5. One exploratory run (validation and test every 64 steps, TensorBoard log)
train --params configs/train/beauty/02_logq_downstream.json

# 6. Confirmatory run: test callback removed, one test evaluation per validation-selected
#    checkpoint, JSON report with sha256 of config, count tables and checkpoints
python scripts/train_confirmatory.py --params configs/train/beauty/02_logq_downstream.json \
    --seed 1 --output results/confirm/beauty_02_logq_downstream_seed1.json

# 7. Queues (used for every table in RESULTS.md Part A): exploratory arms / multi-seed arms
bash scripts/run_queue_generic.sh myqueue beauty 02_logq_downstream 03_graph
bash scripts/run_queue_seeds.sh myseeds beauty 1,2,3 01_orig 02_logq_downstream

# 8. Summaries, checks and figures
python scripts/summarize_runs.py --pattern "beauty_*" --report eval/ndcg@20 eval/recall@1000
python scripts/summarize_seeds.py --prefix beauty
python scripts/check_split_leakage.py --data-dir data/Beauty
python scripts/popularity_baseline.py configs/train/beauty/01_orig.json
python scripts/plot_study_figures.py && python scripts/build_run_index.py
```

Key config knobs: `logq_lambda` (correction strength), `leave_own_out` (q' under
false-negative masking), `scheme: cross_only` (BxB contrastive), `normalize_embeddings`
+ `temperature` (cosine scoring), `seed` / `deterministic`. Graph caches
(`data/*/**.npz`) are keyed by filename and only need deleting when the interaction
data or the graph-building code changes.
