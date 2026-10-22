# General input and evaluation validation

This is a software-contract check, not proof that every dataset or GPU execution
is correct. It supplements the paper audit and the Clothing-specific evidence.
See [paper alignment](PAPER_ALIGNMENT.md) for the equation-by-equation comparison
and which research departures are actually related to logQ.
These fixes apply to the current development code; frozen training snapshots,
historical results and cluster jobs are not migrated.

## Dataset and graph contract

Completed v2 inputs are now checked in both MCLSR and SASRec loaders, not just
when frequency graphs are requested. The six file hashes, catalogue bounds,
raw split alignment/disjointness, raw80/20 evaluation split and exact training
window multiset must agree. Cropping or a history/target intersection must not
hide malformed source rows. V2 SASRec cannot bypass this by selecting an arbitrary
unverified training file. Binary MCLSR graphs also revalidate their source and
loaded windows. Invalid history caps and graph neighborhood sizes fail early.

The legacy reader retains its historical cropping/intersection behavior for
reproduction; it is not advertised as a canonical-event validator. Passing raw
evaluation history/target records as graph rows is unsupported and now produces
an explicit error rather than a missing-key exception.

## Ranking and repeated targets

The evaluation task recommends distinct catalogue items with binary relevance.
Targets are therefore a **set of relevant items**, not a count of repeated
events. For targets `A,A,B`, retrieving `A,B` gives Recall=1 and standard NDCG=1
when they occupy the first two positions. The previous implementation used set
membership for hits but event counts for the denominator, incorrectly lowering
these scores. Only the metric's local relevance set is deduplicated: source
events, training targets, sequence order and graph event counts are preserved.

All four ranking metrics reject duplicate predictions, padding/nonpositive IDs,
inconsistent flat-label lengths, noninteger IDs and invalid cutoffs. Empty target
sets still score zero and user-level macro averaging is unchanged. These metrics
cannot validate an upper catalogue bound on target IDs by themselves; the dataset
loader owns that check.

The historical `comirec-ndcg` normalization remains a separate alternative,
using actual hits for IDCG. It is not silently substituted for standard NDCG,
and exact equivalence to the author's evaluator is not established.

## Inference

MCLSR, SASRec and SASRec-in-batch continue to rank the full real-item catalogue.
Previously seen items remain eligible, so repeated-consumption targets are not
made impossible by a history filter. Labels do not enter the score computation.
Padding and mask IDs are excluded. Invalid `eval_top_k` and NaN/Inf real-item
scores now fail explicitly rather than producing an arbitrary ranking.
For finite scores and valid cutoffs, the previous top-k calculation is retained,
including its existing tie behavior. This change does not replace MCLSR's
attention pooling with SASRec's Transformer or change graph/loss architecture.

## Negative sampling and loss masks

Explicit uniform/popularity samplers and the MCLSR fallback check whether enough
eligible distinct negatives exist. Rejection has a finite budget, followed by
sampling from the remaining eligible support; impossible requests fail rather
than hanging. The current sample's positive is excluded even if the user was not
present in the sampler's original training rows. These paths are distinct from
in-batch negatives; changing one does not establish the other's exact proposal.

Classic BCE SASRec also retains its usual uniform rejection path, but falls back
to uniform sampling from each unresolved user's exact eligible catalogue. It no
longer treats 64 unlucky draws as proof that no candidate exists, and rechecks the
last draw instead of using a stale rejection mask. A genuinely saturated user
still raises an explicit error. This is a Python sampling exception, not evidence
for the cause of a native SIGSEGV.

Hard exclusions in retrieval and contrastive losses now use negative infinity rather
than a finite `-1e12` sentinel. The finite sentinel could outrank an even smaller
valid logit. Corrected objectives explicitly return zero contribution and zero
gradient for rows without any valid negatives, avoiding both fake comparisons
and undefined all-masked log-sum-exp gradients. Objective variants, same-user
mask choices, temperatures and component weights are not silently switched.
Contrastive cross-only, paper, symmetric and full-catalogue paths are checked
against explicitly enumerated allowed candidates, including extreme finite
dot-product and Euclidean scores. Full-catalogue contrastive positives cannot
refer to padding or mask rows.

## Reproducible checks

- `tests/test_ranking_contract.py`: independent scalar metric oracles over all
  permutations of a three-item catalogue, target sequences of length0..3 with
  repetitions, three cutoffs, variable batches, invalid inputs, and all three
  inference models.
- `tests/test_end_to_end.py`: two additional repeated-event pipelines, for binary
  and event-count graphs. Each prepares v2 data and role counts, performs eight
  CPU optimizer updates with all four losses and downstream logQ, selects a
  validation checkpoint, reloads it, and independently recomputes test metrics
  using Python sets/math rather than production metric classes. Source data
  hashes must remain unchanged.
- `tests/test_graph_event_counts.py` and `tests/test_repeated_event_counts.py`:
  canonical repeated events, overlapping windows, graph weights, cache identity,
  and distinct target/context count definitions.
- `tests/test_dataset_input_contract.py`: canonical/raw input validation,
  catalogue boundaries, split consistency, legal repeated windows, malformed
  sources, source revalidation, and graph top-k boundaries.
- `tests/test_negative_sampling_contract.py`: finite sampling on saturated
  catalogues, positive exclusion, exhausted proposal support, exact hard masks,
  and finite/zero gradients without eligible negatives.
- `tests/test_sasrec_negative_contract.py`: forced rejection through the final
  attempt, exact eligible-set fallback, separate per-user exclusions, and genuine
  catalogue saturation for classic BCE SASRec.
- `tests/test_masked_contrastive_contract.py`: hard exclusions, singleton and
  repeated-ID batches, value/gradient oracles and full-catalogue mask boundaries.
- `tests/test_optimizer_contract.py`: a single global clipping norm over all
  optimizer parameter groups, including learned uncertainty-loss weights and
  excluding unoptimized parameters. Previously only model parameters were clipped;
  the fixed-weight component screen does not use the affected combination.

For exact historical source identity, keep the frozen-screen fingerprint test
strict. Modified development sources are not byte-identical to that screen;
its failure must not be relabeled as a successful reproduction. Do not overwrite
old hashes or scientific results to make that check green.

## CPU/CUDA numerical comparison

`scripts/check_device_parity.py --suite base --device cuda:0 --report base.json`
compares one synthetic float32 forward/backward for MCLSR and SASRec-in-batch.
`--suite logq` adds five matched MCLSR cases: duplicate-masked auxiliary Q0,
IL-only, UC-only, IC-only and all auxiliary corrections. Downstream Q remains
enabled in every case. The count fixtures distinguish user-row, target and
context-inclusion roles; repeated user/item IDs exercise the masks.

Each case compares model outputs, component/total losses and every model-parameter
gradient against CPU. Reports include component values and source/config hashes.
The elementwise tolerance is `atol=1e-4, rtol=1e-3`; IDs must match exactly.
Missing gradients, nonfinite values or mismatches are failures, not skips.
Use `--device cpu` for a verifier self-check, not as evidence of CUDA behavior.

These probes disable dropout and TF32, explicitly move graph tensors and route
legacy device-global helpers to each device. They test neither optimizer updates
nor real-data training, arbitrary device migration or long-run native stability.
The numerical tests of independent equations remain separate: agreement between
two devices alone would not prove that the shared formula is correct.

The 17 October TITAN RTX probe was **not fully green**: five of 275 elementwise
tensor checks failed across four MCLSR cases, on shared-interest-projector bias
gradients. SASRec-in-batch passed. The original reports and tolerances were retained.
For the base MCLSR and matched masked-Q0 cases, a diagnostic using identical
initial weights found CPU64/CUDA64 agreement within `1e-9 + 1e-8*abs(reference)`;
CUDA32 also passed against CPU64 at the original float32 bounds. CPU32 did not.
This identifies a numerical-precision issue in these cases, not a demonstrated
CUDA or logQ formula defect. It is not a long-training or SIGSEGV result, and
production activations/dtypes were not changed to hide the mismatch.

## Limits that tests do not remove

- Marginal logQ is not automatically the exact anchor-conditional proposal after
  same-user/item masking. `negatives_only` also changes a constant margin under
  uniform Q. The numerical oracles test the implemented objectives, not a theorem
  that correction must improve every component or dataset.
- Binary and event-frequency graph weights are separate choices; the latter is
  opt-in. See [graph event counts](GRAPH_EVENT_COUNTS.md).
- This CPU pass does not establish the cause of the separate native CUDA SIGSEGV,
  long-run training stability, statistically significant quality improvements,
  or complete reproduction of unavailable author code.
