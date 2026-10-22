# MCLSR: paper alignment and research departures

This is a specification audit of the implemented model and the archived
15-arm component screen, not a claim to reproduce the authors' executable or
published metrics. The reference is Wang et al., *Multi-level Contrastive
Learning Framework for Sequential Recommendation*, CIKM 2022,
[arXiv:2208.13007v1](https://arxiv.org/abs/2208.13007).
The inspected PDF has SHA256
`1fb27b09a64a924dfaa0e4126c54cbe204423f12bfcbff2b361a10d23d8c2eef`.
Equation and section references below identify the paper, not an inferred author
implementation. No verified author source/split/evaluator is available in this audit.

## Verdict

The model implements the principal MCLSR attention, graph, projection, fusion and
inference equations. **The complete experiment is not identical to the paper.**
Some departures are deliberate research choices; others are retained implementation
choices or unresolved author details. Calling every difference a code bug, or
claiming every difference is necessary for logQ, would both be incorrect.

Numerical tests compare values and gradients to independent equations. They do
not prove equivalence of sampling distributions, training dynamics or final quality.
Confirmed software-contract bugs and repairs are documented separately in
[general validation](GENERIC_VALIDATION.md).

## What is actually configured

All 15 archived component-screen configs were read and checked against their
individual manifest hashes. They share the same model, dataset, optimizer and
dataloaders. The profile is:

| Setting | Effective value |
|---|---|
| Model | MCLSR, all three graphs enabled, embedding dimension 64 |
| Graph propagation | 2 layers, final layer, train-only graphs, row-wise top-50 co-action neighbors |
| Graph weights | Binary user-item presence; `event_count` is not enabled in the archived screen |
| Current-interest input | Reverse/recency positions, LayerNorm enabled, input dropout 0.3 |
| Graph regularization | Edge dropout 0.3 |
| Interest projection | One shared IL projector |
| Fusion | `0.5 * current_interest + 0.5 * general_interest`, before projection |
| Downstream | Raw dot, temperature 1, in-batch, `negatives_only`, same-user/same-target exclusions |
| Auxiliary losses | Cosine, temperature 0.5, `scheme='paper'`; coefficients IL=1, UC=0.05, IC=0.05 when present |
| Optimizer/batch | Adam, learning rate 0.001; 128 shuffled rows, drop incomplete train batch |
| Data | `windowed-v2`, up to 20 history items plus one target; user-disjoint splits |
| Selection | Best validation NDCG@20 within 65,536 updates, validation every 256; screen does not evaluate test metrics |
| Inference | Full real-item catalogue, top-1000; reported validation NDCG@20 and Recall@1000 |

This profile is **not the library defaults**. In particular, the library's default
separate IL projectors and symmetric contrastive scheme must not be described as
the archived screen's shared projectors and paper-direction losses. Old dot-based
experiments are a different family and cannot establish the cosine family's outcome.

## Equation-by-equation mapping

| Paper | Implemented computation | Boundary of the match |
|---|---|---|
| Section 3.2 | `GraphDataset`: user-item incidence, user-user and item-item co-action products | Binary default differs from event counts when genuine events repeat; diagonal/pruning choices below |
| Eq. 1, 4, 9 | `MCLSRModel._apply_graph_encoder`: repeated multiplication by normalized adjacency | Normalization matches; dropout is an extension, not part of Eq. 1 |
| Eq. 2 | `softmax(W2 tanh(W1 E_positioned))`, width 4d, one scalar attention weight per valid position | LayerNorm/dropout/reverse indexing alter the inputs relative to the literal formula |
| Eq. 3 | Attention-weighted **raw item embeddings**, not positioned or projected values | This is intentional and matches the displayed formula |
| Eq. 5, 6 | User-conditioned attention over user-item graph item embeddings and its weighted readout | Values and gradients covered, including unequal history lengths and repeats |
| Eq. 7 | Shared two-layer ELU IL projector in the screen | The optional independent-projector mode is a different architecture |
| Eq. 8 | First view anchors; paired other-view positive; all other-view candidates plus other same-view rows | Paper-direction cosine form matches; row identity, masks and reductions need separate qualification |
| Eq. 10, 11 | Shared projector across two views within each entity type, then corresponding contrastive objective | User and item feature losses have separate entity-type projectors; item domain is unique batch-context items |
| Eq. 12, 15 | Weighted sum `LP + IL + 0.05*UC + 0.05*IC` in full arms | Nominal coefficients match; separate means over different domains do not establish the author's effective scaling |
| Eq. 13 | Fusion of the two **pre-projection** interests | Projection is used for alignment, not silently substituted into retrieval |
| Eq. 14 | Dot-product prediction with a sampled denominator | In-batch denominator and optional logQ are research departures, detailed below |
| Eq. 16 | Full-catalogue dot ranking from current interest only | Graphs and projection heads are deliberately not evaluated at inference; this matches the paper |

Core implementations: [model](../src/irec/models/mclsr.py),
[graphs](../src/irec/dataset/graph.py),
[contrastive losses](../src/irec/loss/contrastive.py),
[retrieval losses](../src/irec/loss/retrieval.py),
[composition](../src/irec/loss/base.py).

## Departures: which are for logQ, and which are not?

| Departure or uncertainty | Why it is present / what is established | Required by logQ? |
|---|---|---|
| In-batch candidates rather than paper's 1280 sampled negatives | Deliberate controlled study of in-batch correction. Batch 128 supplies at most 127 negative columns before masking, not 1280; dq0 and dq1 share this sampling protocol. | No mathematical requirement: logQ can also be studied with explicit sampling. This is the selected research protocol. |
| Negative-only `-lambda*log Q`; positive uncorrected | Explicit objective choice. The printed MCLSR equations do not specify it; the author's underlying sampled-softmax library behavior is unknown. | This is the correction being studied, not a repair needed to implement MCLSR. |
| Same-user/same-target masking; auxiliary duplicate-ID masking in masked arms | Avoid selected false negatives. It changes the retained candidate distribution and must be held fixed across Q comparisons. Plain `fps` masks row self, not all duplicate entities. | No. Masking and correction are independent interventions. |
| Binary presence instead of repeated-event frequency | Preserves historical baseline. Opt-in canonical `event_count` now implements frequency weighting; overlapping windows are not independent events. | No. See [graph weights](GRAPH_EVENT_COUNTS.md). |
| Co-action diagonal removed; top-50 selected separately per row | Explicit implementation choices; row-wise pruning can produce asymmetric adjacency. Author self-loop/tie/symmetrization policy is unavailable. | No. Do not silently change them as a logQ fix. |
| Input LayerNorm, input/edge dropout 0.3 | Extra normalization/regularization retained by the screen. The initial choice is not justified by the paper; a switch for unregularized arms exists. | No. Their necessity or optimality has not been established. |
| Reverse position indexing | Learned recency positions rather than the displayed forward indexing. Present implementation choice, not proven author behavior. | No. |
| Per-loss means and unique IC items | IL/UC average over batch rows; IC averages over unique context items. Paper writes sums and does not fully specify minibatch reduction/domain. Equal coefficient labels alone do not imply equal effective weighting. | No. |
| History cap 20 **plus** target | Explicit v2 next-item contract; preserves all target positions. Paper's "sample length 20" does not identify whether target is included. | No. It fixes a data-contract ambiguity, not logQ. |
| Data filtering/ties and exact split IDs | V2 uses the documented iterative core and deterministic split; article prose does not completely identify its preprocessing. Clothing totals differ by one real item from Table 1; exact author source/tie policy unknown. | No. Do not alter data merely to force a table match. |
| Standard set-relevance metrics and validation selection | Precisely defined locally, including repeated targets. Exact author NDCG normalization, stopping/seeds and tie behavior are not recoverable from the supplied paper alone. | No. |

Disabling all auxiliary **losses** does not remove the general-interest branch
from training retrieval. Disabling **all graphs** leaves MCLSR attention pooling,
not SASRec's causal Transformer. The paper's graph ablation using initial node
embeddings is also not automatically equivalent to `use_graph=False`.
These are separate questions and should use separately named controls.

## LogQ is not automatically an exact correction

The downstream table models marginal target-row frequencies. With row shuffling
without replacement and same-user exclusions, the remaining negative law depends
on the anchor. For a non-positive item j, its expected retained multiplicity is

`(B - 1) / (N - 1) * (C_j - C_user,j)`.

It is not simply the marginal `C_j/N`. Leave-positive-out renormalization does
not account for the entire same-user mask. For IC, `1-(1-p_context)^B` is an IID
marginal inclusion proxy; actual shuffled batches and anchor co-occurrence do
not make it an exact conditional inclusion probability.

Moreover, negative-only correction also changes a constant margin:

`-lambda*log(Q_j) = -lambda*log(V*Q_j) + lambda*log(V)`.

The last term does not cancel because the positive is uncorrected. Even uniform Q
changes the loss: with zero scores, two negatives and V=10, `log(3)` becomes
`log(21)` at lambda=1. Consequently a gain cannot be attributed solely to removing
popularity bias, nor does a loss decrease establish better recommendation quality.

The five masked auxiliary arms isolate correction coefficients while retaining
the same masks. Compare them against `masked_dq1_auxq0`, not directly against
`full_dq1`, which also differs in duplicate masking. See the exact
[15-arm matrix](component-screen.md) and [matched seed follow-up](logq-followup.md).

## Evidence and limits

- `test_paper_architecture_oracle.py`: NumPy forward equations and finite-difference
  parameter gradients; readouts, sharing, fusion, graph operators and reductions.
- `test_paper_downstream_oracle.py`: analytic value/gradient checks, exhaustive
  conditional-sampling enumeration and margin decomposition.
- `test_paper_protocol_oracle.py`: inference, selection, data protocol and strict
  archived-source identity. The identity assertion intentionally detects changed
  development sources; it must not be loosened to claim old-run reproduction.
- `test_contrastive_logq_audit.py`, `test_retrieval_logq_audit.py`: candidate domains,
  probability roles, variants, masks and numerical gradients.
- [General validation](GENERIC_VALIDATION.md): repeated events through preparation,
  graph/count construction, CPU training, checkpoint reload and independent metrics.

The full local suite rerun on 22 October with the original evidence bundle
has 1097 passes and one archived-source-identity failure.
This is not a fully green suite or a claim
that frozen results were recomputed. Source/config/metric changes must be frozen
in a new experiment snapshot before their quality can be compared scientifically.

The archive-profile assertions are separate from source identity: all 15 archived
configs and their declared loss interventions are checked even when the old-source
fingerprint assertion fails. Global gradient clipping now also covers learned
loss weights through the optimizer's parameter groups; this conditional bug did
not affect the fixed-weight component screen.

For a portable checkout, archive-dependent tests use an explicit
`MCLSR_SCREEN_EVIDENCE` path rather than a machine-local default. Without that
external bundle they are reported as skipped, not validated. Providing the
original bundle retains the strict source-identity failure described above;
neither the historical hashes nor the identity assertion have been changed.
The portable run on 22 October has 1078 passes and 20 archive-dependent skips;
lint and format checks pass for all source, script and test files.

No test here proves GPU stability or resolves the separate native SIGSEGV.
No scientific configs, archived results or running jobs are modified by this
document. A verified author implementation or clarification is still needed for
the unresolved author choices; new runs are needed for claims about quality.
