
# Steps on local machine
<p align="center">
   <a>
      <img src="./assets/irec_logo.png" alt="cool irec logo" width="40%" height="40%">
   </a>
</p>

<p align="center">
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
All pre-processed datasets used in our experiments are available for download from our cloud storage. This is the fastest way to get started.

- **link**

After downloading, extract the archives into the [data](./data) directory.

Alternatively, you can process the raw data from scratch using the Jupyter notebooks provided in the [notebooks](./notebooks) directory. The notebooks will generate the required .txt data splits.

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

# 4. Loss correctness tests (reference implementations)
python tests/test_logq_losses.py

# 5. Training (the maintained grid lives in configs/train/grid/)
train --params configs/train/grid/03_graph.json

# 6. Results summary across runs
python scripts/summarize_runs.py
```

Key config knobs: `logq_lambda` (correction strength), `leave_own_out` (q' under
false-negative masking), `scheme: cross_only` (BxB contrastive), `normalize_embeddings`
+ `temperature` (cosine scoring), `seed` / `deterministic`. Graph caches
(`data/*/**.npz`) are keyed by filename and only need deleting when the interaction
data or the graph-building code changes.
