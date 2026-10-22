# Data protocols and compatibility

## Protocols

Keep historical datasets, count tables, checkpoints and reports separate from new protocol runs.
Changing preprocessing requires a new dataset directory and experiment namespace.

| Contract | `legacy-v1` | `windowed-v2` |
|---|---|---|
| Preparation CLI | default for historical reproduction | explicit `--protocol windowed-v2` |
| Training MCLSR targets | targets from the last 20 events only | every event after the first |
| Input history | at most 19 after removing the target | at most `max_history_length` preceding events |
| Evaluation history | at most 20 by historical default | the same history capacity as training |
| IDs | preserved, including unsafe real item 0 in some old files | users/items remapped densely to 1..N; 0 reserved |
| Metadata | no preprocessing manifest | `preprocessing.json`: mappings, dimensions, history capacity, file hashes |
| SASRec weighting | final cropped sequence, all-position loss | final sequence with one extra event so input capacity matches evaluation; still not MCLSR's all-target weighting |

For 25 ordered events and history capacity 20, the windowed MCLSR ladder has 24 rows, not 19.
Each row contains at most 20 context events **plus** its target. The model's length remains 20.
Keeping every target changes the training protocol, including the relative weight of long-history
users and the train-only graphs. Truncating a user's full history is not inherently a bug: it can
be an intentional protocol choice. V2 aligns train/evaluation history capacity, but its quality
and fidelity to the paper must be assessed separately; more examples do not prove either.
The dataset loader recognizes the manifest automatically and rejects conflicting length settings.
It also checks the six file hashes and refuses an incomplete preprocessing directory.
V2 catalogue dimensions come from the full manifest, not a cropped training history; legacy
datasets retain the historical post-crop dimension rules for checkpoint compatibility.
An explicit `max_history_length` controls row lengths but does not prove v2 preprocessing and
cannot recover events removed by old preprocessing. Set `expected_protocol: "windowed-v2"`
in the inner dataset config to require a validated preprocessing manifest.
SASRec remains a SASRec-style post-LN backbone, not an exact copy
of the original TensorFlow architecture.

## Prepare a new dataset without changing the old one

Use the original, uncropped chronological sequences or raw ratings, never an old cropped
`train_sasrec.txt` as though it were the full source. The sequence input must already satisfy the
documented 5-core assumptions; CSV input performs iterative filtering.

```bash
python scripts/prepare_dataset.py --seq data/Beauty/Beauty.txt \
  --protocol windowed-v2 --max-history-length 20 --out data/Beauty_v2
```

The output directory must be new or empty. For CSV use `--csv` instead of `--seq`.
Read `num_items` from the new manifest, not from an old configuration or dataset name.
Generate a new target-count table for MCLSR in-batch retrieval using row length **21**, not 20:

```bash
NUM_ITEMS=$(python -c 'import json; print(json.load(open("data/Beauty_v2/preprocessing.json"))["num_items"])')
python scripts/generate_mclsr_role_counts.py --input data/Beauty_v2/train_mclsr.txt \
  --num_items "$NUM_ITEMS" --max_len 21 \
  --target_output data/Beauty_v2/item_target_counts.pkl \
  --context_output data/Beauty_v2/item_context_counts.pkl
```

The count generator infers this row length from the v2 manifest when `--max_len` is omitted,
and rejects an explicit conflicting value. Without a manifest its default stays legacy 20.

Copy a configuration under a **new experiment name**. Set its nested dataset name to `Beauty_v2`,
its graph directory to `data/Beauty_v2`, and the retrieval count path to the new target artifact.
Use `expected_counts_role: "target"` and `expected_counts_max_len: 21` for that retrieval loss.
The SASRec all-position sampler needs its own target proposal; do not blindly reuse MCLSR counts.
For a correction-only comparison keep variant, masks, batch size, graph/regularization settings,
data and checkpoint selection fixed and change only lambda. Changing `corrected`, Q and same-user
masking together is a combined intervention, not an isolated correction ablation.

## Input and provenance requirements

- Graph archives embed a canonical interaction hash, dimensions, graph type, options and builder
  version; missing/mismatched/corrupt provenance triggers a rebuild. Writes replace one complete
  archive atomically. Legacy filenames remain readable by SciPy and the verifier.
- Dataset `--verify-against` and mandatory architecture/graph checks return nonzero on failure.
  Reproducing legacy files confirms that preprocessing procedure, not paper fidelity.
  The standalone architecture/graph verifiers are legacy-protocol checks and explicitly refuse
  v2; use the v2 sampler/count/end-to-end regression tests for the new data contract.
- Paper-logQ numerical oracles use catalogue IDs independently of batch positions and check
  gradients with nonuniform frequencies and arbitrary IDs.
- Paired bootstrap aligns unique user IDs across arms/seeds, reports seed differences, and labels
  its uncertainty as conditional on the trained seeds. Both metrics use NDCG-selected checkpoints.
  It verifies config/checkpoint hashes before scoring, and checks recorded input hashes when a
  provenance sidecar exists. Legacy reports without a sidecar cannot certify their original data.
- Confirmatory runs record source/data/config/count/environment provenance **before** training,
  and refuse collisions with existing run outputs/checkpoint namespaces. Hash inventories identify
  dirty source but are not a restorable archive; retain the exact source snapshot separately.
  Imported `irec` modules must originate in the inventoried source tree; a stale editable install
  fails instead of recording the new checkout while training the old code.

Run `python -m pytest -q tests` before training. Keep each GPU run in a separate working directory
or use unique experiment/seed names. Do not edit a source tree while experiments are consuming it.

## Frozen protocol comparison

Prepare five MCLSR arms on the same executable: legacy without logQ, legacy with historical
proxy Q, legacy with target Q, windowed-v2 without logQ, and windowed-v2 with target Q. Each arm
uses seeds 1, 2 and 3. This compares data/proposal protocols, not the historical executable.

```bash
python scripts/prepare_protocol_study.py --csv /absolute/path/to/Clothing/data.csv \
  --legacy-reference /absolute/path/to/historical/Clothing \
  --output /absolute/path/to/new/protocol_ab \
  --template configs/train/clothing64/05_full_paper.json \
  --steps 250000 --validation-every 256
python scripts/run_protocol_study.py --study /absolute/path/to/new/protocol_ab/study.json --worker 0
python scripts/run_protocol_study.py --study /absolute/path/to/new/protocol_ab/study.json --worker 1
```

Run the two workers in separate GPU allocations, each with its own visible GPU. Preparation and
training belong on compute nodes, not the cluster login/controller node. Preparation requires
byte-identical reconstruction of the six legacy files, logically identical evaluation rows and
the same positive dense candidate catalogue. It refuses to overwrite an existing study.

Each run has the same actual optimizer-update budget, not equal epochs or early-stopping patience.
Validation runs at the common interval (including the training loop's initial step-zero callback).
Only validation NDCG@20 selects a checkpoint; test NDCG@20 and Recall@1000 are scored once afterward.
The source, configurations and prepared inputs are hashed before training and checked before each
run. Each queue reserves its outputs exclusively and stops on the first failed run; no implicit
resume occurs. Keep the source tree immutable while either worker is running.

Target Q is a marginal ladder-target frequency, not the exact anchor-conditional proposal after
sampling without replacement and applying masks. The v2 comparison jointly changes target coverage,
maximum training context and train-only graphs; it does not isolate those three mechanisms.

### Recovering an interrupted queue

The default runner stops on a failed child. `scripts/recover_protocol_study.py` provides an explicit
recovery namespace without overwriting the original source, manifests, logs or checkpoints. For a
previously frozen checkout, copy this supervisor **outside** its `src/` and `scripts/` directories
and put the frozen checkout's `src` and `scripts` on `PYTHONPATH` when invoking the external copy:

```bash
PYTHONPATH=/absolute/frozen/repo/src:/absolute/frozen/repo/scripts \
  python /absolute/artifacts/recover_protocol_study.py \
  --study /absolute/frozen/repo/studies/protocol_ab/study.json \
  --recovery recovery_01 --worker 0
```

Start worker 1 separately on the second allocated GPU. Original queues must no longer be running.
The supervisor validates original inputs, reserves each worker exclusively and records its own hash
and recovery policy. It runs unstarted trials first, then failed original trials from their original
seeds. Best-model weights are not an exact training resume: optimizer/RNG/sampler state is absent.
Attempt configurations change only the experiment namespace and recovery-manifest reference.

A classified CUDA launch/illegal-address failure or SIGSEGV before training completion can get one
bounded retry after a fresh-process GPU probe. Failed original trials have already used their first
attempt. Invalid data, code, budgets, report hashes or an unhealthy GPU halt recovery. Two newly
classified infrastructure failures across workers prevent further launches; already running children
may finish. A successful short GPU probe does not establish the cause of a previous CUDA failure.
Runs that may already have reached test evaluation require manual reconciliation, not another training
attempt/test score. All failed attempts remain visible; do not choose among successful repeats by score.

A shared stop prevents new launches, not acceptance of an already completed child. The supervisor
still checks input integrity and the completed report before recording `complete`. Older supervisors
may leave `running` followed by `halted` despite a valid saved report; reconcile the report and
checkpoint hashes before considering any repeat.

### Continuing after a halted recovery

After investigating the failure and confirming the old workers have exited, use an explicit new
namespace and `--previous-recovery /absolute/path/to/old/recovery`. The supervisor reconciles both
the original queues and the first recovery. A valid report is retained even when the old event log
ends in `running`, then `halted`. It validates the original scientific configuration, provenance,
checkpoint and update budget, and pins old evidence without changing it or deleting the old HALT.

```bash
PYTHONPATH=/absolute/frozen/repo/src:/absolute/frozen/repo/scripts \
  python /absolute/artifacts/recover_protocol_study.py \
  --study /absolute/frozen/repo/studies/protocol_ab/study.json \
  --recovery continuation_02 --previous-recovery /absolute/path/to/recovery_01 \
  --worker 0 --launcher /absolute/artifacts/native-launcher.sh
```

Run worker 1 on the other allocated GPU with identical arguments except `--worker 1`.
Each worker runs its entire remaining partition sequentially, not just its first trial. Finished
trials are skipped; unstarted trials precede previously failed trials. Original and prior recovery
attempts share a two-attempt limit. Ambiguous possibly-tested runs, invalid reports and exhausted
budgets block continuation. This mode accepts a first-generation recovery, not recursive recovery
chains. Diagnostic runs without test evaluation are not scientific attempts or results.

An optional executable `--launcher` is prepended to each training command and its hash is guarded.
It can invoke a native debugger with a bounded timeout; it must preserve child exit status and isolate
debugger-only library paths. Record and keep runtime settings consistent across the new workers.
The first new failure prevents further launches on both queues; an already running healthy child
may finish and have its report recorded. There are **no automatic retries** in continuation mode.
This is a controlled continuation after diagnosis, not a claim that the underlying crash is fixed.

### Diagnosing native crashes without repeating test

`scripts/diagnose_training.py` reuses the frozen confirmatory runner and exits immediately after
bounded training, before the test dataloader is constructed. It records a separate diagnostic
manifest and result, and must not be included among confirmatory study results. Copy the launcher
outside an already frozen source tree, then run it within a compute allocation:

```bash
PYTHONPATH=/absolute/frozen/repo/src:/absolute/frozen/repo/scripts \
  PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 \
  python /absolute/artifacts/diagnose_training.py \
  --study /absolute/frozen/repo/studies/protocol_ab/study.json \
  --arm legacy_target_l10 --seed 1 --steps 50000 \
  --label native_gpu0 --output-dir /absolute/artifacts/diagnostic_native_gpu0
```

Use a fresh label and output directory for every attempt. The original configuration is copied
with only a diagnostic experiment name; the smaller update budget is explicit in the diagnostic
manifest and effective run provenance. Best weights may be saved, but no test report is produced.
Run under a native debugger to capture a SIGSEGV stack; an isolated follow-up with
`CUDA_LAUNCH_BLOCKING=1` can help locate asynchronous CUDA errors. Neither a short successful
probe nor disappearance of the error under a debugger/synchronization establishes a fix.

## What remains a research question

Cosine similarity belongs in contrastive Eqs. 8/11; dot-product retrieval in Eqs. 14/16 is correct.
`*_paper` configurations correct the contrastive scheme/projector, but still differ in
regularization, sampling and preprocessing. `*_paper_raw` removes regularization from a graph+IL
ablation, not from the complete feature-level model. Neither name guarantees full reproduction.

Finite sampled logQ gradients are not generally exact/unbiased full-softmax gradients. Smoothing
Q does not give sampling support to never-sampled items. Row-multiplicity proposals and unique-item
inclusion probabilities must be derived separately. A complete paper-reference run, sampler-matched
counts, all-target SASRec comparison, seed-level uncertainty, original dataset identity and original
NDCG normalization still require explicit verification. Do not infer them from a green unit suite.
