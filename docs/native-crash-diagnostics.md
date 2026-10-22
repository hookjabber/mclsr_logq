# Native loss diagnostics

A successful short probe does not establish that long training is stable. Keep
diagnostics separate from study reports, and do not resume from a best-weights
checkpoint as if it contained optimizer, sampler and RNG state.

## Isolate cross-entropy

Inside an existing compute-node allocation, using the same Python environment as
the failing run:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout --signal=INT --kill-after=30s 22m python \
  scripts/trace_cross_entropy.py --trace-file /tmp/ce-probe.trace.json \
  scripts/stress_cross_entropy.py --output-dir /tmp/ce-probe \
  --device cuda --seed 3 --iterations 1000000 \
  --max-seconds 1200 --progress-every 10000
```

Use fresh paths on every invocation. The probe refuses an existing output
directory; the trace refuses an existing file. CUDA never silently falls back to
CPU. Tiny synthetic CPU tests can explicitly use `--device cpu --batch-size 3`.

One iteration performs CE forward/backward on float32 `[128,128]` and `[128,256]`
scores with diagonal int64 labels. `--item-sizes 1 512 2560` adds item-like
`[U,2U]` shapes. These are synthetic tensors, not real model states; the probe does
not load datasets, a model or checkpoints. It checks finite loss/gradients and
synchronizes more often than training, so a pass cannot rule out an asynchronous
interaction with sparse operations.

`result.json` exists only when all requested iterations finish. A time limit,
Python exception, native signal or interruption is not a successful probe. The
Python time budget is cooperative; the external `timeout` bounds a hung native
call. Progress and environment metadata are in `events.jsonl` and `metadata.json`.
Boundary logging there is sampled, not a record of every call.

## Capture the last actual CE call

The trace wrapper keeps a fixed 16 KiB mmap-backed JSON file. It records each CE
entry and return, tensor shape/stride/dtype/device, Python call stack, and (when
called from the training loop) the loss name and zero-based optimizer step.
It does not read CUDA tensor values or change arguments, outputs or gradients.
The wrapper changes timing and is intended for a single-threaded diagnostic
caller; it is not a scientific experimental arm.

`phase=enter` means that CE did not return before the snapshot. `phase=returned`
does **not** mean asynchronous GPU work completed. Reading the file during a
write can produce a partial JSON snapshot; prefer reading after process exit.
The mmap is not fsynced per call and is not a node-crash-proof journal.

For the full training graph, wrap `scripts/diagnose_training.py` instead of the
stress script. Its existing frozen-input guards and training-only budget remain
active, and it exits before constructing the test dataloader. Keep the wrapper
and diagnostic outputs outside a frozen source snapshot. Do not point the trace
wrapper at the ordinary scientific runner and assume test evaluation is disabled.

## Native debugger evidence

Python tracebacks alone may identify the reporting call, not the original fault.
Capture the stop signal and `$_siginfo`, frame-zero registers, live instruction
bytes, caller frames/stack memory and loaded-library mappings before the process
is terminated. Preserve the original child exit status. Check that debugger-only
libraries do not leak into the Python child. An on-disk wheel checksum does not
prove that runtime code or registers were intact.
