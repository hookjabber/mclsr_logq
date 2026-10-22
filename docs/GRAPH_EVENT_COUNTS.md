# Repeated events and graph weights

## Two different graph definitions

The default `interaction_weighting: "binary"` uses presence weights:
`R[user, item] = 1` if the training data contains that pair. Multiple genuine
events and repeated occurrences of an event across training windows both collapse
to a single edge. This is a presence graph, not an event-frequency graph.

The opt-in `interaction_weighting: "event_count"` counts genuine events:
for a training sequence `A, B, A`, the weights are `R[user, A] = 2` and
`R[user, B] = 1`. The windows `A,B` and `A,B,A` must **not** be counted together
as five independent events. Changing only the window history cap must not change graph weights.

For either mode, user-user weights are `R @ R.T` and item-item weights are
`R.T @ R`, followed by the existing diagonal removal, optional row-wise top-k,
and degree normalization. Frequency weights change all three graphs, not just
user-item weights. Top-k ties and asymmetry retain the existing policy.

Section 3.2 of arXiv:2208.13007v1 describes user-item edge weights as the number
of interactions. `event_count` implements this weighting. It does not certify
the remaining architecture, self-loop policy, evaluation protocol, or an exact
reproduction of the authors' implementation.

## Selecting frequency weights

The explicit event-count profile is
`configs/train/validation/clothing_v2_event_count_l00.json`. Its matched control is
`configs/train/validation/clothing_v2_binary_l00.json`. They differ **only** in
`interaction_weighting` and experiment name.
Both inner datasets explicitly set `expected_protocol: "windowed-v2"`, requiring a validated
preprocessing manifest. A directory name alone is not accepted as proof of v2 preprocessing.
Legacy configs without this requirement continue to support manifest-free inputs.

Both profiles use all three graphs, v2 history capacity 20, shared IL projection,
cosine/paper auxiliary losses at temperature 0.5 and weights 1/0.05/0.05. Downstream
is in-batch with logQ disabled; auxiliary logQ is also disabled. LayerNorm and input/edge
dropout 0.3 are retained so graph weighting is the only comparison factor. These are
graph controls within the in-batch study, **not** full paper-replication profiles.
They use seed 1, 65,536 updates and validation every 256 updates, selecting NDCG@20;
no test-evaluation callback is configured. One seed is diagnostic, not confirmation.

Before running either profile, prepare completed `windowed-v2` data in the separate
`data/Clothing_v2` directory from the original uncropped source, as described in
[the v2 instructions](PROTOCOL_V2.md). Generate `item_target_counts.pkl` and
`item_context_counts.pkl` with `generate_mclsr_role_counts.py`, using the new manifest's
catalogue size and row cap 21. The target artifact is required even at logQ lambda zero.
Do not substitute graph-event frequencies for target/context proposals.

Run each profile in a separate experiment namespace without loading an old checkpoint:

```bash
PYTHONPATH=src python -m irec.train \
  --params configs/train/validation/clothing_v2_event_count_l00.json
```

Use the companion binary configuration for a matched comparison. These validation-only
profiles are not inputs to `train_confirmatory.py`, which requires a test-evaluation callback.
If train sequences have no repeated user-item events, the graph modes coincide;
running both then introduces no distinct graph intervention.

Add this field to the **graph dataset wrapper**, not the inner sequence dataset:

```json
{
  "type": "graph",
  "interaction_weighting": "event_count",
  "use_train_data_only": true
}
```

Keep the wrapper's other fields (`dataset`, `graph_dir_path`, graph flags and
neighborhood size). The inner dataset must be `type: "mclsr"` backed by completed
`windowed-v2` preprocessing. No existing experiment configuration is switched
automatically; an absent field still means `binary`.

Frequency weights use the full `train_sasrec.txt` sequences retained by v2, even
though the graph's training sampler consumes bounded `train_mclsr.txt` windows.
The loader verifies all six source hashes and raw rows before cropping or
history/target intersections can hide invalid data: valid non-padding IDs, one
sequence per training user, matching unique history/target users, disjoint splits,
and exact agreement with the **multiset** of training windows. Identical
legitimate windows retain their multiplicity.
Truncated/missing/extra windows and inconsistent event sequences fail explicitly.
This raw v2 input validation also applies to binary MCLSR and SASRec loaders;
it is not restricted to frequency graphs. For evaluation, inference and sampling
contracts, see [general validation](GENERIC_VALIDATION.md).

Legacy data, datasets without a canonical event source, and
`use_train_data_only: false` are not supported by `event_count`; the code refuses
to invent frequencies from arbitrary cropped or overlapping rows. Convert from
the original sequences using v2 preprocessing in a new directory instead.

## Cache and experiment isolation

Binary filenames stay unchanged. Frequency caches contain `_event_count` in the
filename and cannot overwrite binary caches. Manifests identify the mode and
builder version. The frequency digest includes sorted `(user, item, count)`
triples, so changing multiplicity with unchanged support invalidates every graph;
reordering events/windows or changing the history cap with unchanged pair counts
does not change the graph identity. Old manifests are rebuilt on next use.

Record the selected weighting with each experiment; do not
compare different modes as if the only changed factor were logQ.

## Checks independent of a particular dataset

`tests/test_graph_event_counts.py` uses synthetic repeated events, exact dense
incidence/co-action oracles before and after normalization, all length-four
sequences over a two-item alphabet, history caps, cache mutation/reuse, isolated
nodes, and invalid-source rejection. It does not rely on Clothing having unique
user-item pairs.

Graph weights and logQ tables are different objects. Target counts count target
rows; context-inclusion counts count an item once per row context; user counts
count training rows. Those definitions must not be replaced wholesale by graph
event frequencies. Repeated-event count tests live in
`tests/test_repeated_event_counts.py`.
