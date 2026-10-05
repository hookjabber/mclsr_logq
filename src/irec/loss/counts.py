"""Count tables behind the logQ correction: loading, role / provenance metadata, validation."""

import os
import pickle


def load_counts_artifact(path, expected_role=None, expected_max_len=None):
    """Load a counts table: either a bare array (legacy) or a metadata dict
    {'counts', 'denominator', 'role', 'max_len', 'source', 'source_sha256'}.
    Returns (counts_array, metadata_dict); metadata is {} for bare arrays.

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
    return counts, metadata
