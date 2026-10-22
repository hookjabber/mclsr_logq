"""Record the latest CE boundary in a bounded file without reading CUDA tensor data.

This single-process diagnostic changes timing, not the loss computation. It is
not a scientific result and does not establish CUDA operation completion.
"""

import argparse
from contextlib import contextmanager
from functools import wraps
import json
import mmap
import os
from pathlib import Path
import runpy
import sys

import torch.nn.functional as functional

TRACE_BYTES = 16384


def tensor_metadata(tensor):
    if tensor is None:
        return None
    return {
        'shape': list(tensor.shape),
        'stride': list(tensor.stride()),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'storage_offset': tensor.storage_offset(),
    }


def call_context(frame):
    context = {'python_stack': [], 'loss': None, 'optimizer_step_zero_based': None}
    try:
        for _ in range(16):
            if frame is None:
                break
            context['python_stack'].append(
                {'file': frame.f_code.co_filename, 'line': frame.f_lineno, 'function': frame.f_code.co_name}
            )
            instance = frame.f_locals.get('self')
            prefix = getattr(instance, '_output_prefix', None)
            if context['loss'] is None and isinstance(prefix, str):
                context['loss'] = prefix
            if frame.f_globals.get('__name__') == 'irec.training' and frame.f_code.co_name == 'train':
                context['optimizer_step_zero_based'] = frame.f_locals.get('step_num')
            frame = frame.f_back
        return context
    finally:
        del frame


@contextmanager
def trace_cross_entropy(path):
    original = functional.cross_entropy
    with Path(path).open('x+b') as output:
        output.truncate(TRACE_BYTES)
        with mmap.mmap(output.fileno(), TRACE_BYTES, access=mmap.ACCESS_WRITE) as trace:
            calls = 0

            def record(state):
                encoded = json.dumps(state, sort_keys=True).encode()
                if len(encoded) >= TRACE_BYTES:
                    raise RuntimeError('CE trace exceeds its fixed storage budget')
                trace[:] = encoded.ljust(TRACE_BYTES - 1, b' ') + b'\n'

            record({'phase': 'ready', 'pid': os.getpid(), 'call': 0})

            @wraps(original)
            def traced(*args, **kwargs):
                nonlocal calls
                calls += 1

                def argument(position, name, default=None):
                    return args[position] if len(args) > position else kwargs.get(name, default)

                state = {
                    'phase': 'enter',
                    'pid': os.getpid(),
                    'call': calls,
                    'input': tensor_metadata(argument(0, 'input')),
                    'target': tensor_metadata(argument(1, 'target')),
                    'weight': tensor_metadata(argument(2, 'weight')),
                    'size_average': argument(3, 'size_average'),
                    'ignore_index': argument(4, 'ignore_index', -100),
                    'reduce': argument(5, 'reduce'),
                    'reduction': argument(6, 'reduction', 'mean'),
                    'label_smoothing': argument(7, 'label_smoothing', 0.0),
                    **call_context(sys._getframe(1)),
                }
                record(state)
                result = original(*args, **kwargs)
                state['phase'] = 'returned'
                record(state)
                return result

            functional.cross_entropy = traced
            try:
                yield
            finally:
                functional.cross_entropy = original
                trace.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-file', type=Path, required=True)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original_argv = sys.argv
    try:
        sys.argv = [str(args.script), *args.arguments]
        with trace_cross_entropy(args.trace_file):
            runpy.run_path(str(args.script), run_name='__main__')
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
