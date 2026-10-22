"""Build role-exact logQ counts from an MCLSR ladder.

With a sibling windowed-v2 preprocessing.json, infer the row cap as max_history_length + 1
(history plus target). Without a manifest, retain the legacy default row cap of 20.
An explicit --max_len must agree with the v2 manifest; legacy inputs allow an override.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def _resolve_max_len(input_path, max_len):
    if max_len is not None and max_len < 1:
        raise ValueError('--max_len must be a positive integer')
    data_dir = Path(input_path).parent
    if (data_dir / '.preprocessing-in-progress').exists():
        raise ValueError('dataset preprocessing is incomplete')
    manifest_path = data_dir / 'preprocessing.json'
    if not manifest_path.exists():
        return 20 if max_len is None else max_len
    with manifest_path.open() as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or manifest.get('protocol') != 'windowed-v2':
        raise ValueError('unsupported preprocessing manifest protocol')
    history_length = manifest.get('max_history_length')
    if type(history_length) is not int or history_length < 1:
        raise ValueError('preprocessing manifest requires a positive integer max_history_length')
    row_length = history_length + 1
    if max_len is not None and max_len != row_length:
        raise ValueError(
            f'--max_len={max_len} conflicts with windowed-v2 preprocessing manifest: '
            f'max_history_length={history_length} requires --max_len={row_length} (history plus target)'
        )
    return row_length


def main():
    parser = argparse.ArgumentParser(
        description='Role-exact count tables for logQ from the mclsr ladder file. '
        'item_counts.pkl counts every event in full sequences — a proxy for both '
        'logQ uses. The exact proposals are: TARGET counts (how often an item is '
        'the last element of a ladder line = the L_P positive distribution) and '
        'CONTEXT-INCLUSION counts (in how many ladder line inputs an item occurs '
        '= the L_IC in-batch candidate distribution; unique per line, truncation '
        'applied exactly as in the train sampler).',
    )
    parser.add_argument('--input', required=True, help='Path to train_mclsr.txt')
    parser.add_argument('--num_items', type=int, required=True)
    parser.add_argument(
        '--max_len',
        type=int,
        help='Row cap including the target, not just history. Inferred as max_history_length + 1 '
        'from a sibling windowed-v2 preprocessing.json (history 20 means 21); an explicit value '
        'must match that manifest. Without a manifest, defaults to legacy 20 and allows an override.',
    )
    parser.add_argument('--target_output', required=True)
    parser.add_argument('--context_output', required=True)
    args = parser.parse_args()
    try:
        args.max_len = _resolve_max_len(args.input, args.max_len)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    size = args.num_items + 2
    target_counts = np.zeros(size, dtype=np.float32)
    context_counts = np.zeros(size, dtype=np.float32)

    lines = 0
    with open(args.input) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            item_ids = [int(i) for i in parts[1:]][-args.max_len :]
            target_counts[item_ids[-1]] += 1
            for item_id in set(item_ids[:-1]):
                context_counts[item_id] += 1
            lines += 1

    with open(args.input, 'rb') as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()

    # counts keep RAW ZEROS for absent ids (the numerical clamp lives in the
    # loss); the artifact is a self-describing dict so a loss can validate the
    # denominator/semantics instead of trusting hand-copied config numbers
    for role, counts, denominator, path in (
        ('target', target_counts, float(target_counts.sum()), args.target_output),
        ('context-inclusion', context_counts, float(lines), args.context_output),
    ):
        artifact = {
            'counts': counts,
            'denominator': denominator,
            'role': role,
            'max_len': args.max_len,
            'num_lines': lines,
            'source': args.input,
            'source_sha256': source_sha256,
        }
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)
        print(
            f'{role}: {lines} lines, sum={int(counts.sum())}, '
            f'zeros={int((counts == 0).sum())}, denominator={denominator:.0f} '
            f'-> {path}'
        )


if __name__ == '__main__':
    main()
