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

The commands in this section reproduce the **historical `legacy-v1`** datasets and tables.
For new experiments use [the versioned `windowed-v2` recipe](docs/PROTOCOL_V2.md) in a new
directory. Do not mix remapped IDs, new windows/counts or checkpoints with historical artifacts.

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
Archives without matching embedded input provenance are rebuilt; copy legacy artifacts to a
separate run directory if the original archive bytes need to be preserved.

Graph weights default to binary user-item presence, including on datasets with repeated
events. For explicit canonical event-frequency weights, see
[repeated events and graph weights](docs/GRAPH_EVENT_COUNTS.md). This opt-in v2 mode has
separate caches and does not change existing experiment configurations.

## 3. Checks before training

See [general validation](docs/GENERIC_VALIDATION.md) for repeated-event semantics,
input validation, sampling boundaries and the independent CPU checks. The strict
frozen-screen source-identity test intentionally fails when development sources
differ from the archived run; do not update its expected hashes to hide that drift.
Archive-dependent tests require the original external evidence bundle, which is
not distributed with this repository. They are explicitly skipped when
`MCLSR_SCREEN_EVIDENCE` is unset; this does not certify historical source identity.
Set it to the original bundle's path to run those checks. An explicitly supplied
missing file is an error, and changed source fingerprints still fail:

```bash
MCLSR_SCREEN_EVIDENCE=/path/to/component-screen-results-evidence-20261015.tgz \
  python -m pytest -q tests/test_paper_protocol_oracle.py
```

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
Historical reports describe their original runs, not later development fixes.
Compare source fingerprints where available before attributing old results to
the current checkout; a passing numerical test does not prove source identity.
