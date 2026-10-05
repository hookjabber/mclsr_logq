# Reproducing the study end to end

Everything below runs from the repository root. A CPU is enough for the data, the checks and the
tests; a single GPU trains one Beauty arm in about an hour.

## 1. Environment

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv sync --frozen                       # or: pip install -e .
export PYTHONPATH=src
```

## 2. Data

Raw Amazon ratings (`user,item,rating,timestamp`, no header) go to `data/<Name>/data.csv`.

```bash
# split files: 5-core, dense ids, chronological order, users 8:1:1 with seed 42, train cropped to
# 20 items and unrolled into a prefix ladder, eval users cut at int(0.8 * len)
python scripts/prepare_dataset.py --csv data/Clothing/data.csv --out data/Clothing
# a dataset that already comes as one chronological 5-core sequence per user:
python scripts/prepare_dataset.py --seq data/Beauty/Beauty.txt --out data/Beauty

# count tables for the correction (num_items = the largest item id)
python scripts/generate_item_counts.py --input data/Clothing/train_sasrec.txt \
    --output data/Clothing/item_counts.pkl --num_items 23033
# optional tables for the role-exact and matched-anchor arms
python scripts/generate_mclsr_role_counts.py --input data/Clothing/train_mclsr.txt --num_items 23033 \
    --max_len 20 --target_output data/Clothing/item_target_counts.pkl --context_output data/Clothing/item_context_counts.pkl
python scripts/generate_train_presence.py --mode item --input data/Clothing/train_sasrec.txt \
    --output data/Clothing/train_presence_items.pkl --num_entities 23033
python scripts/generate_user_counts.py --input data/Clothing/train_mclsr.txt --output data/Clothing/user_counts.pkl

python scripts/check_split_leakage.py --data-dir data/Clothing
```

The graphs (`*.npz`) are built on the first run of a graph config and cached next to the data.

## 3. Checks before training

```bash
python -m pytest -q tests                                   # reference, config, runtime, end-to-end
python scripts/verify_graphs.py Clothing                    # after the first graph run
python scripts/indep_check_architecture.py --config configs/train/clothing64/05_full_l00.json --data data/Clothing
python scripts/indep_check_sasrec.py --config configs/train/clothing64/sasrec_inbatch_l00.json --mclsr-config configs/train/clothing64/01_orig.json
```

## 4. Training

```bash
# one exploratory run: validation and test every 64 steps, TensorBoard under tensorboard_logs/
python -m irec.train --params configs/train/beauty/02_logq_downstream.json

# the confirmatory protocol used for every number in RESULTS.md Part A: test callback removed,
# one test read per validation-selected checkpoint, JSON report with sha256 of everything
python scripts/train_confirmatory.py --params configs/train/beauty/02_logq_downstream.json \
    --seed 1 --output results/confirm/beauty_02_logq_downstream_seed1.json

# queues: arms x seeds (done-markers in results/, logs in results/logs/)
bash scripts/run_queue_seeds.sh headline beauty 1,2,3 01_orig 02_logq_downstream 03_graph 05_full_baseline
```

## 5. Numbers, uncertainty, figures

```bash
python scripts/summarize_seeds.py --prefix beauty --pairs 02_logq_downstream:01_orig 04_graph_logq_lil:03_graph
python scripts/paired_bootstrap.py --prefix beauty --config-dir configs/train/beauty --seeds 1 2 3 \
    --pairs 02_logq_downstream:01_orig 04_graph_logq_lil:03_graph --output results/stats/beauty_paired_bootstrap.md
python scripts/eval_checkpoint.py --help                     # score any checkpoint on validation / test
python scripts/plot_study_figures.py && python scripts/plot_paired_bootstrap.py
python scripts/build_run_index.py                            # RUN_INDEX.md
```

## What a result rests on

A number in [RESULTS.md](./RESULTS.md) Part A is the mean over three seeds of a test metric read
once, on the checkpoint selected by validation ndcg@20; its report in `results/confirm/` carries the
sha256 of the config, the count tables and the checkpoint, the git commit and the torch version.
[VERIFICATION.md](./VERIFICATION.md) lists the independent checks behind every stage.
The `git_commit` field of a report names the commit of the cluster working tree the run was
started from; that clone carries an older history of the same code (the public history was
rewritten before release), so the field dates a run rather than pointing into this repository.
The code behind every report is the code verified by the tests and the checks above.
