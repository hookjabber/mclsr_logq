# Validation-only component and logQ screen

`scripts/run_component_screen.py` separates removal of an auxiliary objective,
false-negative masking, and logQ on each objective. This is an exploratory
single-seed screen, not a replacement for the frozen multi-seed study. It does
not evaluate the test split. Dataset construction and provenance still read
test files, but the runner exits before building its test dataloader.

## Comparisons

All arms derive from the existing study's `v2_l00.json`: identical data, graphs,
architecture, cosine/paper contrastive directions, loss weights, optimizer and
batching. Each starts from the same seed. The default budget is **65,536 updates**,
not the original study's 250,000; do not compare different budgets as matched arms.

An optional `--source-root` selects a separately copied source snapshot with only
the audited count-validation and stable-inclusion repairs. Its entire inventory
is frozen too; added/deleted files or any other source differences are rejected.
All screen arms use that same snapshot, not a mixture of old/new executables.

| Arms | Purpose |
|---|---|
| `full_dq0`, `full_dq1` | Original FPS objectives, downstream logQ off/on |
| `no_interest_dq0/1` | Remove interest alignment objective |
| `no_user_feature_dq0/1` | Remove user-feature objective |
| `no_item_feature_dq0/1` | Remove item-feature objective |
| `downstream_only_dq0/1` | Remove all three auxiliary objectives |
| `masked_dq1_auxq0` | All auxiliary objectives use FPS-logQ with lambda zero and duplicate masking |
| `masked_dq1_interest_q1` | Add interest logQ to the matched masked control |
| `masked_dq1_user_feature_q1` | Add user-feature logQ to the same control |
| `masked_dq1_item_feature_q1` | Add item-feature logQ to the same control |
| `masked_dq1_all_q1` | Add all three auxiliary corrections to the same control |

Removed objectives are actually absent, not merely assigned zero weight. The
model still computes its graph branches: this is a **loss ablation**, not an
architecture ablation. `downstream_only` is still MCLSR, not SASRec.

Compare auxiliary-logQ arms primarily with `masked_dq1_auxq0`, not `full_dq1`.
The latter comparison also changes duplicate-user masking. Component removal
changes the total training objective; it does not retune remaining loss weights.

## What Q means here

- Downstream: target-row frequencies; `negatives_only`, dot, temperature 1,
  same-user/item masking inherited from the frozen implementation. No claim of
  exact anchor-conditional sampling or finite-sample unbiased gradients.
- Interest/user-feature: user frequencies in the **v2 training ladder**, with
  zero counts preserved for absent IDs. `sample` mode respects the fact that
  candidate rows retain multiplicity. Leave-own-out is disabled.
- Item-feature: per-history inclusion counts, transformed as
  `1 - (1 - p_context)^batch_rows`, using the number of training **rows**, not
  unique items or events. This is an **IID marginal inclusion proxy**. Actual
  shuffling samples without replacement, and anchor co-occurrence is not captured.
- Paired positives are not auxiliary-Q corrected. Consequently uncentered Q
  also changes the negative-versus-positive margin, even for uniform Q. This
  screen does not distinguish that margin effect from popularity reweighting;
  centered/constant-Q controls would be a separate follow-up.

## Run beside a frozen study

Do not add tooling to an already frozen source tree: its source inventory must
remain unchanged. Put this script and `diagnose_training.py` in an external
tool directory. Use the same Python environment as the study and set
`PYTHONPATH` to its `src` directory. The CLI derives the remaining source paths
from `study.json`.

```bash
python /path/to/tools/run_component_screen.py prepare \
  --study /path/to/study/study.json --output-dir /path/to/component_screen \
  --steps 65536 --seed 1
python /path/to/tools/run_component_screen.py queue \
  --manifest /path/to/component_screen/screen.json
```

Run training only on an allocated compute node. An optional `prepare --launcher`
accepts a Bash wrapper that receives the entire child command as arguments;
the wrapper must propagate its real exit status. This allows a native debugger
and timeout without changing the model source.

Preparation freezes config, tooling and user-count hashes and links to the
original study hash. The queue verifies these and the original source/data
before every arm. Each arm is a fresh Python process. Selected validation
checkpoints and TensorBoard logs live in the execution snapshot's ordinary output
directories with unique screen names; JSON reports/logs live under the screen.

Read `queue.events.jsonl`, `logs/<arm>.log`, and `runs/<arm>/result.json`.
Successful reports contain exact update count, best/last validation metadata,
checkpoint and provenance hashes, and `test_evaluated: false`.

No implicit retries or overwrites are allowed. An isolated SIGSEGV/SIGBUS is
recorded as a failed arm and the queue advances. Two consecutive or three total
native failures stop it; any other error or changed input stops immediately.
Failed/incomplete arms remain visible and are not ranked as completed trials.
`queue.summary.json` lists failed and unattempted arms. Even if the queue reaches
its end, any failed arm yields a nonzero exit status. A new attempt requires a
new namespace and an explicitly recorded decision, never deletion of a failure.

Best validation scores after a short, single-seed screen suggest follow-ups;
they cannot establish significance, paper equivalence, or final test quality.

The [matched user-feature logQ follow-up](logq-followup.md) prepares the masked
control and UC-Q variant on seeds 2 and 3 without launching training or changing
the frozen source. Paper-alignment and native-crash diagnostics remain separate.
