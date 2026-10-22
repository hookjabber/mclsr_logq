# Tests and numerical references

## Local tests

Use Python 3.12 and the locked development dependencies:

```bash
uv sync --frozen --extra dev
export PYTHONPATH=src
uv run --frozen --extra dev python -m pytest -q -rs tests
uv run --frozen --extra dev ruff check src scripts tests
uv run --frozen --extra dev ruff format --check src scripts tests
```

| Area | Reference |
|---|---|
| Sequence protocols, source caches and v2 requirements | `tests/test_sequence_protocol.py`, `tests/test_protocol_reconciliation.py`, `tests/test_validation_graph_profiles.py` |
| Binary/event-count graphs, normalization and sparse indices | `tests/test_graph_event_counts.py`, `tests/test_graph_reconciliation.py` |
| MCLSR equations and parameter gradients | `tests/test_paper_architecture_oracle.py`, `tests/test_model_reconciliation.py` |
| Loss values, gradients, masks and count normalization | `tests/test_logq_losses.py`, `tests/test_loss_reconciliation.py`, `tests/test_negative_sampling_contract.py` |
| Catalogue ranking and repeated targets | `tests/test_ranking_contract.py` |
| Training, checkpoint selection/reload and independent metrics | `tests/test_end_to_end.py`, `tests/test_runtime.py` |

Archive-specific tests require `MCLSR_SCREEN_EVIDENCE`. They are skipped without the
external archive. Source-fingerprint checks require the exact historical source;
a modified checkout cannot certify the original executable.

The [data protocols](docs/PROTOCOL_V2.md), [graph weights](docs/GRAPH_EVENT_COUNTS.md)
and [paper comparison](docs/PAPER_ALIGNMENT.md) define the supported contracts and
scientific limitations. A passing test suite is not proof of complete paper
reproduction, a quality improvement or long-run GPU stability.

## Real-data tools

- `scripts/prepare_dataset.py --verify-against`: reconstruct and compare split files.
- `scripts/verify_graphs.py`: compare legacy binary graph caches with independent SciPy construction.
- `scripts/indep_check_architecture.py`: NumPy forward and loss references for the legacy study profile.
- `scripts/indep_check_sasrec.py`: NumPy causal-transformer and training-query references.
- `scripts/check_device_parity.py`: bounded CPU/device output and gradient comparisons.
- `scripts/eval_checkpoint.py`: score a saved checkpoint under its configured evaluation protocol.
- `scripts/paired_bootstrap.py`: paired user-bootstrap intervals conditional on trained seeds.

The standalone legacy graph/architecture tools reject v2 inputs. Use the v2
regression tests for canonical-window and repeated-event contracts. Reproduction
commands and dataset requirements are in [REPRODUCE.md](REPRODUCE.md).

## LogQ objective definitions

For a query with positive item p, retained in-batch negatives N, scores s and a
configured marginal table q(v), `mclsr_logq_inbatch` implements:

| Form | Configuration | Per-query loss |
|---|---|---|
| No correction | `logq_lambda: 0` | −s_p + log(e^{s_p} + Σ_{j∈N} e^{s_j}) |
| Negatives-only | `variant: negatives_only`, λ | −s_p + log(e^{s_p} + Σ_j e^{s_j − λ log q(j)}) |
| Standard | `variant: standard` | −(s_p − λ log q(p)) + log(e^{s_p − λ log q(p)} + Σ_j e^{s_j − λ log q(j)}) |
| Corrected | `variant: corrected` | sg(1 − P̂) · (−s_p + log Σ_j e^{s_j − λ log Q′(j)}), Q′(j) = q(j) / (1 − q(p)), P̂ = e^{s_p} / (e^{s_p} + mean_j e^{s_j − λ log Q′(j)}) |
| Leave-positive-out | `leave_own_out: true` | Negatives-only with q′(j) = q(j) / (1 − q(p)) |
| Mixed sampling | `mixed_uniform_negatives: K` | K shared uniform items; configured mixture B/(B+K)·q + K/(B+K)/|V| |

These are definitions of implemented objectives, not a claim that each marginal
table is the exact conditional proposal after same-user/item masks or item-union
deduplication. Negative-only correction also changes the negative-positive margin
under uniform q. See [paper alignment](docs/PAPER_ALIGNMENT.md#logq-is-not-automatically-an-exact-correction).
