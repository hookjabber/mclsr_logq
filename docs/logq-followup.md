# Matched user-feature logQ seed follow-up

This is a follow-up to the validation-only component screen, not a new
paper-replication configuration. It checks whether the small seed-1 user-feature
Q improvement repeats on seeds **2 and 3**. It does not repeat all 15 arms or
claim that Q improves every component.

Each seed has two sequential arms:

| Arm | Downstream Q | Interest Q | User-feature Q | Item-feature Q |
|---|---:|---:|---:|---:|
| `masked_dq1_auxq0` | 1 | 0 | 0 | 0 |
| `masked_dq1_user_feature_q1` | 1 | 0 | 1 | 0 |

All losses and model graph branches remain enabled. Both arms use identical
duplicate masks, source snapshot, data/count artifacts, architecture, loss
weights, similarity, temperature, optimizer, and batch size. The preparer
requires the two parent configurations to differ only in experiment name and
the user-feature Q coefficient. New configs change only experiment names;
seeds are supplied by the queue manifests.

The budget remains **65,536 updates**, validation every **256**, with checkpoint
selection by `validation/ndcg@20`. Test evaluation is disabled. The two seed-1
parent results must be complete and pass the existing result/hash validation.

## Prepare, without starting training

Keep the preparer outside the frozen source tree on the cluster. Use the
existing frozen screen runner; do not replace its code or native launcher.

```bash
python /path/to/tools/prepare_logq_followup.py \
  --parent /path/to/component_screen/screen.json \
  --parent-sha256 EXPECTED_PARENT_SHA256 \
  --screen-tool /path/to/frozen_tools/run_component_screen.py \
  --output-dir /path/to/logq_followup
```

Preparation creates `followup.json`, `seed_2/screen.json`, `seed_3/screen.json`,
and four config files. It refuses existing output/checkpoint/event namespaces.
It preserves parent hashes, source/data/tool guards, and result provenance.
`followup.json` contains the exact queue command for each seed.

**Preparation does not submit Slurm jobs, start background workers, reserve a
queue, modify tmux, or schedule training after an existing job.** Run the recorded
commands only after explicitly checking an idle GPU allocation. The existing
native launcher requires its original compute-node environment. Each seed's
control and UC-Q arm run sequentially; two seed queues may use two independently
allocated GPUs. Keep each pair on the same GPU where practical.

The existing runner preserves failed arms and does not retry or overwrite them.
It advances after an isolated native SIGSEGV/SIGBUS but stops after two consecutive
native failures, three total native failures, any other error, or changed inputs.
A failed pair is incomplete, not a usable result. Inspect `queue.events.jsonl`,
`queue.summary.json`, per-arm logs, and validated `result.json` files; a shell
prompt or a Slurm `RUNNING` allocation alone does not establish completion.

## Interpret separately from the paper audit

Report the paired UC-Q minus control delta for each seed, then the mean and
spread across seeds 1–3. Compare Recall@1000 at the NDCG-selected checkpoint,
not a separately optimized step. Do not choose the best seed, substitute partial
failed runs, or treat three seeds as proof of a universal improvement.

This comparison still combines popularity reweighting with the uncentered
positive/negative margin effect described in `component-screen.md`. Distinguishing
those mechanisms needs separate centered/constant-Q controls. Changes needed for
paper alignment, graph-branch ablations, SASRec equivalence, and native-crash
diagnostics are separate experiments, not changes to these frozen pairs.
