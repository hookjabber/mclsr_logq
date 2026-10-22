"""Count tables behind the logQ correction: loading, role / provenance metadata, validation."""

import os
import pickle

import numpy as np


def counts_to_probabilities(counts, denominator=None):
    count_array = np.asarray(counts, dtype=np.float64)
    if denominator is None:
        denominator = count_array.sum()
    denominator_array = np.asarray(denominator)
    if (
        denominator_array.ndim != 0
        or denominator_array.dtype.kind not in 'iuf'
        or not np.isfinite(denominator_array)
        or denominator_array <= 0
        or (count_array > denominator_array).any()
    ):
        raise ValueError('Invalid counts denominator: expected a positive finite scalar at least every count')
    return np.clip(count_array / denominator_array, 1e-10, 1.0).astype(np.float32)


def load_counts_artifact(path, expected_role=None, expected_max_len=None):
    """Load a counts table: either a bare array (legacy) or a metadata dict
    {'counts', 'denominator', 'role', 'max_len', 'source', 'source_sha256'}.
    Returns (counts, metadata_dict), preserving the stored counts type and dtype;
    metadata is {} for bare arrays. Counts must be a nonempty real numeric vector
    with finite nonnegative values and a positive finite total. An optional
    denominator must be a positive finite scalar no smaller than any count.

    When the config states expectations, they are ENFORCED: a table with the
    wrong role/max_len (e.g. target counts wired where context-inclusion Q
    belongs — both may share a denominator by coincidence) fails loudly, and
    expectations against a legacy bare array fail too (it cannot prove them)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Counts file not found at {path}')
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if isinstance(artifact, dict):
        metadata = {k: v for k, v in artifact.items() if k != 'counts'}
        counts = artifact['counts']
    else:
        metadata, counts = {}, artifact
    if expected_role is not None and metadata.get('role') != expected_role:
        raise ValueError(
            'counts artifact role mismatch at {}: expected {!r}, got {!r}'.format(
                path,
                expected_role,
                metadata.get('role'),
            )
        )
    if expected_max_len is not None and metadata.get('max_len') != expected_max_len:
        raise ValueError(
            'counts artifact max_len mismatch at {}: expected {}, got {}'.format(
                path,
                expected_max_len,
                metadata.get('max_len'),
            )
        )
    try:
        count_array = np.asarray(counts)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid counts at {path}: expected a nonempty real numeric vector') from error
    if count_array.ndim != 1 or count_array.size == 0 or count_array.dtype.kind not in 'iuf':
        raise ValueError(f'Invalid counts at {path}: expected a nonempty real numeric vector')
    if not np.isfinite(count_array).all() or (count_array < 0).any():
        raise ValueError(f'Invalid counts at {path}: values must be finite and nonnegative')
    with np.errstate(over='ignore', invalid='ignore'):
        total = count_array.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f'Invalid counts at {path}: total must be positive and finite')
    if 'denominator' in metadata:
        try:
            denominator = np.asarray(metadata['denominator'])
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid counts denominator at {path}: expected a positive finite real scalar') from error
        if (
            denominator.ndim != 0
            or denominator.dtype.kind not in 'iuf'
            or not np.isfinite(denominator)
            or denominator <= 0
        ):
            raise ValueError(f'Invalid counts denominator at {path}: expected a positive finite real scalar')
        if (count_array > denominator).any():
            raise ValueError(f'Invalid counts denominator at {path}: must be at least every count')
    return counts, metadata
