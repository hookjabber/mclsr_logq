
# Steps on local machine
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

## Repository layout

| path | what |
|---|---|
| `configs/train/grid/` | the maintained experiment grid (numbered, one question per config) |
| `configs/train/legacy/` | historical configs kept for provenance |
| `src/irec/` | framework: models, losses, datasets, metrics, callbacks |
| `scripts/` | count-table generation, run summarization, checkpoint evaluation |
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
[data](./data), then generate the count tables and masks as shown in the
reproduce section below.

## Model training
To train a model, simply run the following from the root directory:
```shell
train --params /path/to/config
```

The script has 1 input argument: `params` which is the path to the json file with model configuration. The example of such file can be found [here](./configs). This directory contrains json files with model hyperparameters and data preparation instructions. It should contain the following keys:

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

# 4. Loss correctness tests (reference implementations)
python tests/test_logq_losses.py

# 5. Training (the maintained grid lives in configs/train/grid/)
train --params configs/train/grid/03_graph.json

# 6. Results summary across runs (fresh runs land in ./tensorboard_logs;
#    in this repository's own history the post-June runs live in
#    new_tensorboard_logs — pass --logs new_tensorboard_logs to summarize them)
python scripts/summarize_runs.py
```

Key config knobs: `logq_lambda` (correction strength), `leave_own_out` (q' under
false-negative masking), `scheme: cross_only` (BxB contrastive), `normalize_embeddings`
+ `temperature` (cosine scoring), `seed` / `deterministic`. Graph caches
(`data/*/**.npz`) are keyed by filename and only need deleting when the interaction
data or the graph-building code changes.
