"""Synthetic CE forward/backward probe; no model, dataset, or checkpoint access.

An iteration visits every requested shape. Time limits are checked between calls;
use an external process timeout to bound a hung native/CUDA call. A completed run
only establishes that this workload passed, not that the training graph is safe.
Boundary logs are sampled; the last recorded call may lag a native crash.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import torch
from torch.nn import functional


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are refused.')
    parser.add_argument('--device', default='cuda', help='cuda, cuda:N, or explicit cpu; never falls back.')
    parser.add_argument('--iterations', type=int, default=100, help='Full shape rounds, 1..1000000.')
    parser.add_argument('--max-seconds', type=float, default=60, help='Cooperative deadline, >0 and <=28800 seconds.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128, help='1..128; use tiny values for CPU smoke tests.')
    parser.add_argument(
        '--item-sizes', type=int, nargs='+', default=[], help='Optional U values, 1..2560; adds U x 2U.'
    )
    parser.add_argument('--progress-every', type=int, default=100, help='Print progress every N completed rounds.')
    args = parser.parse_args(argv)
    for name, upper in [('iterations', 1_000_000), ('batch_size', 128), ('progress_every', 1_000_000)]:
        if not 1 <= getattr(args, name) <= upper:
            parser.error(f'{name} must be in 1..{upper}')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 28_800:
        parser.error('max-seconds must be finite, positive and <=28800')
    if not 0 <= args.seed < 2**63:
        parser.error('seed must be in 0..2**63-1')
    if any(not 1 <= size <= 2560 for size in args.item_sizes) or len(args.item_sizes) > 16:
        parser.error('item-sizes accepts at most 16 values, each in 1..2560')
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device):
        parser.error('device must be cpu, cuda, or cuda:N')
    return args


def device_metadata(device):
    if device.type == 'cpu':
        return {'device': 'cpu', 'device_name': platform.machine(), 'device_uuid': None}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, 'uuid', None)
    return {
        'device': f'cuda:{index}',
        'device_name': properties.name,
        'device_uuid': str(uuid) if uuid is not None else None,
        'device_capability': [properties.major, properties.minor],
        'device_total_memory': properties.total_memory,
    }


def write_json(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed_calls = 0
    completed_iterations = 0
    shapes = [[args.batch_size, args.batch_size], [args.batch_size, 2 * args.batch_size]]
    shapes.extend([size, 2 * size] for size in args.item_sizes)
    metadata = {
        'kind': 'synthetic_cross_entropy_diagnostic',
        'torch_version': str(torch.__version__),
        'torch_git_version': torch.version.git_version,
        'cuda_version': torch.version.cuda,
        'python_version': platform.python_version(),
        'requested_device': args.device,
        'seed': args.seed,
        'iterations': args.iterations,
        'max_seconds': args.max_seconds,
        'shapes': shapes,
        'dtype': 'torch.float32',
        'reduction': 'mean',
        'weight': None,
        'progress_every': args.progress_every,
        'boundary_recording': 'first, progress and final rounds; last recorded call may lag',
        'torch_threads': torch.get_num_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING'),
    }
    with (args.output_dir / 'events.jsonl').open('x') as events:

        def emit(event, *, display=False, **fields):
            payload = {'event': event, 'elapsed_seconds': time.monotonic() - started, **fields}
            line = json.dumps(payload, sort_keys=True, allow_nan=False)
            events.write(line + '\n')
            events.flush()
            if display:
                print(line, flush=True)

        def finish(status, error=None):
            result = {
                'kind': metadata['kind'],
                'status': status,
                'completed_iterations': completed_iterations,
                'completed_calls': completed_calls,
                'elapsed_seconds': time.monotonic() - started,
            }
            if error is not None:
                result['error'] = error
            if status == 'completed':
                write_json(args.output_dir / 'result.json', result)
            emit('terminal', display=True, **result)
            return {'completed': 0, 'time_limit': 124, 'interrupted': 130}.get(status, 1)

        emit('requested', display=True, **metadata)
        try:
            metadata.update(device_metadata(torch.device(args.device)))
            write_json(args.output_dir / 'metadata.json', metadata)
            emit('metadata', display=True, **metadata)
            device = torch.device(metadata['device'])
            generator = torch.Generator(device=device).manual_seed(args.seed)
            for iteration in range(1, args.iterations + 1):
                progress_round = iteration % args.progress_every == 0 or iteration == args.iterations
                for rows, columns in shapes:
                    if time.monotonic() - started >= args.max_seconds:
                        return finish('time_limit')
                    scores = torch.randn(
                        (rows, columns), dtype=torch.float32, device=device, generator=generator, requires_grad=True
                    )
                    targets = torch.arange(rows, device=device, dtype=torch.int64)
                    if iteration == 1 or progress_round:
                        emit(
                            'before_ce',
                            iteration=iteration,
                            call=completed_calls + 1,
                            shape=list(scores.shape),
                            stride=list(scores.stride()),
                            dtype=str(scores.dtype),
                            target_shape=list(targets.shape),
                            target_stride=list(targets.stride()),
                            target_dtype=str(targets.dtype),
                            device=str(scores.device),
                        )
                    loss = functional.cross_entropy(scores, targets)
                    loss.backward()
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError('nonfinite loss')
                    if scores.grad is None or not torch.isfinite(scores.grad).all().item():
                        raise FloatingPointError('missing or nonfinite gradient')
                    completed_calls += 1
                    del loss, scores, targets
                    if time.monotonic() - started >= args.max_seconds:
                        return finish('time_limit')
                completed_iterations = iteration
                if progress_round:
                    emit(
                        'progress',
                        display=True,
                        completed_iterations=iteration,
                        completed_calls=completed_calls,
                        loss=loss_value,
                    )
            return finish('completed')
        except KeyboardInterrupt:
            return finish('interrupted', 'KeyboardInterrupt')
        except Exception as error:
            return finish('error', f'{type(error).__name__}: {error}')


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except OSError as error:
        print(json.dumps({'event': 'output_error', 'error': str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
