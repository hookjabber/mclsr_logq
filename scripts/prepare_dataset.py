#!/usr/bin/env python
"""Rebuild the dataset files of the study from raw data, reproducibly.

Two inputs:
* ``--csv``  an Amazon ratings file (``user,item,rating,timestamp``, no header): the steps of
  ``notebooks/AmazonClothingDatasetUserSplit.ipynb`` -- iterative 5-core filtering on users and
  items, dense-rank ids (1..N in sorted order of the raw ids), rows sorted by (user, timestamp),
  one chronological sequence per user;
* ``--seq``  a text file with one line per user, ``user item item ...``, already 5-core with dense
  ids and in chronological order (how Beauty and CDs & Vinyl were prepared).

The historical ``legacy-v1`` protocol: users are shuffled with ``random.seed(42)``
and split 80/10/10; a training user keeps the last 20 items and contributes a prefix ladder (one
line per target position, ``train_mclsr.txt``) and the cropped sequence (``train_sasrec.txt``); a
validation / test user is split at ``int(0.8 * len)`` into history and targets.

``--protocol windowed-v2`` keeps every MCLSR training target with a bounded preceding history,
maps users/items to positive dense IDs, and records the mapping and output hashes. Write it into
a NEW directory; historical data and checkpoints must not be mixed with this protocol. SASRec's
full user sequences are retained; its loader still trains on each user's final bounded sequence.

``--verify-against DIR`` compares the produced files with existing ones line by line and, where
lines differ, says whether the difference is only a reordering within the line (e.g. a timestamp tie).

    python scripts/prepare_dataset.py --csv data/Clothing/data.csv --out /tmp/clothing_rebuilt \
        --verify-against data/Clothing
"""

import argparse
import os
import random
import hashlib
import json
from contextlib import ExitStack
from itertools import zip_longest

MAX_LEN_TRAIN = 20
SPLIT_SEED = 42


def sequences_from_csv(path):
    import polars as pl

    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=['user_id', 'item_id', 'rating', 'timestamp'],
        separator=',',
        schema_overrides={'user_id': pl.String, 'item_id': pl.String, 'rating': pl.Float64, 'timestamp': pl.UInt64},
    )
    threshold = 5
    filtered = df.clone()
    changed = True
    stage = 0
    while changed:
        user_counts = filtered.group_by('user_id').agg(pl.len().alias('user_count'))
        item_counts = filtered.group_by('item_id').agg(pl.len().alias('item_count'))
        good_users = user_counts.filter(pl.col('user_count') >= threshold).select('user_id')
        good_items = item_counts.filter(pl.col('item_count') >= threshold).select('item_id')
        old_size = len(filtered)
        filtered = filtered.join(good_users, on='user_id', how='inner').join(good_items, on='item_id', how='inner')
        changed = old_size != len(filtered)
        stage += 1
        print(f'5-core stage {stage}: users {good_users.shape[0]}, items {good_items.shape[0]}, rows {len(filtered)}')
    filtered = filtered.with_columns(user_id=pl.col('user_id').rank('dense'))
    filtered = filtered.with_columns(item_id=pl.col('item_id').rank('dense'))
    filtered = filtered.sort(['user_id', 'timestamp'])
    grouped = filtered.group_by('user_id', maintain_order=True).agg(pl.all().exclude('user_id'))
    return [(int(user_id), [int(i) for i in items]) for user_id, items, _, _ in grouped.iter_rows()]


def sequences_from_seq_file(path):
    seqs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            seqs.append((int(parts[0]), [int(x) for x in parts[1:]]))
    uids = [u for u, _ in seqs]
    assert uids == sorted(uids) and len(set(uids)) == len(uids), 'user ids must be ascending and unique'
    assert min(len(s) for _, s in seqs) >= 5, '5-core violated'
    return seqs


def split_users(seqs):
    all_user_ids = [u for u, _ in seqs]
    random.seed(SPLIT_SEED)
    random.shuffle(all_user_ids)
    n = len(all_user_ids)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    return (
        set(all_user_ids[:n_train]),
        set(all_user_ids[n_train : n_train + n_valid]),
        set(all_user_ids[n_train + n_valid :]),
    )


def write_files(seqs, train_u, valid_u, test_u, out, protocol='legacy-v1', max_history_length=20):
    if protocol not in ('legacy-v1', 'windowed-v2'):
        raise ValueError(f'unknown preprocessing protocol: {protocol}')
    if not isinstance(max_history_length, int) or max_history_length < 1:
        raise ValueError('max_history_length must be a positive integer')
    if protocol == 'legacy-v1' and max_history_length != MAX_LEN_TRAIN:
        raise ValueError('legacy-v1 uses its fixed historical length of 20')
    users = [user for user, _ in seqs]
    if len(users) != len(set(users)) or train_u & valid_u or train_u & test_u or valid_u & test_u:
        raise ValueError('user split must be disjoint with unique input users')
    if train_u | valid_u | test_u != set(users):
        raise ValueError('user split must cover exactly the input users')
    if not seqs or any(len(items) < 2 or any(item < 0 for item in items) for _, items in seqs):
        raise ValueError('sequences need at least two nonnegative item IDs')
    if any(len(items) < 5 for user, items in seqs if user not in train_u):
        raise ValueError('evaluation users need at least five items')
    names = ('train_mclsr', 'train_sasrec', 'valid_history', 'valid_target', 'test_history', 'test_target')
    manifest_path = os.path.join(out, 'preprocessing.json')
    if protocol == 'windowed-v2':
        if os.path.isdir(out) and os.listdir(out):
            raise FileExistsError(f'windowed-v2 requires a new or empty output directory: {out}')
        user_mapping = {user: index + 1 for index, user in enumerate(sorted(users))}
        item_mapping = {
            item: index + 1 for index, item in enumerate(sorted({item for _, items in seqs for item in items}))
        }
    elif os.path.exists(manifest_path):
        raise FileExistsError('refusing to overwrite a versioned dataset with legacy-v1')
    os.makedirs(out, exist_ok=True)
    incomplete_path = os.path.join(out, '.preprocessing-in-progress')
    if protocol == 'windowed-v2':
        with open(incomplete_path, 'x') as marker:
            marker.write(protocol + '\n')
    ladder = 0
    with ExitStack() as stack:
        files = {name: stack.enter_context(open(os.path.join(out, name + '.txt'), 'w')) for name in names}
        for original_user, original_items in seqs:
            uid = user_mapping[original_user] if protocol == 'windowed-v2' else original_user
            items = [item_mapping[item] for item in original_items] if protocol == 'windowed-v2' else original_items
            if original_user in train_u:
                crop = items if protocol == 'windowed-v2' else items[-MAX_LEN_TRAIN:]
                files['train_sasrec'].write(' '.join(map(str, [uid] + crop)) + '\n')
                for target_index in range(1, len(crop)):
                    start = max(0, target_index - max_history_length) if protocol == 'windowed-v2' else 0
                    row = [uid] + crop[start : target_index + 1]
                    files['train_mclsr'].write(' '.join(map(str, row)) + '\n')
                    ladder += 1
            else:
                split_index = int(len(items) * 0.8)
                kind = 'valid' if original_user in valid_u else 'test'
                files[kind + '_history'].write(' '.join(map(str, [uid] + items[:split_index])) + '\n')
                files[kind + '_target'].write(' '.join(map(str, [uid] + items[split_index:])) + '\n')
    if protocol == 'windowed-v2':
        hashes = {}
        for name in names:
            digest = hashlib.sha256()
            with open(os.path.join(out, name + '.txt'), 'rb') as source:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    digest.update(chunk)
            hashes[name + '.txt'] = digest.hexdigest()
        manifest = {
            'protocol': protocol,
            'max_history_length': max_history_length,
            'padding_id': 0,
            'num_users': len(user_mapping),
            'num_items': len(item_mapping),
            'train_targets': ladder,
            'user_id_mapping': user_mapping,
            'item_id_mapping': item_mapping,
            'split_seed': SPLIT_SEED,
            'files_sha256': hashes,
            'sasrec_training': 'one final bounded sequence per training user; not all-target MCLSR weighting',
        }
        with open(manifest_path, 'x') as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write('\n')
        os.unlink(incomplete_path)
    return ladder


def verify(out, ref):
    """Line-by-line comparison with the files in ``ref``; a differing line is classified as a
    reordering (same user, same multiset of items -- e.g. items sharing a timestamp) or a real
    difference."""
    names = ('train_mclsr', 'train_sasrec', 'valid_history', 'valid_target', 'test_history', 'test_target')
    all_ok = True
    for n in names:
        a = open(os.path.join(out, n + '.txt')).read().split('\n')
        b = open(os.path.join(ref, n + '.txt')).read().split('\n')
        if a == b:
            print(f'  {n + ".txt":20s} identical ({len(a) - 1} lines)')
            continue
        all_ok = False
        diff = [(x, y) for x, y in zip_longest(a, b, fillvalue='') if x != y]
        reorder = sum(1 for x, y in diff if x.split()[:1] == y.split()[:1] and sorted(x.split()) == sorted(y.split()))
        print(
            f'  {n + ".txt":20s} DIFFERS: {len(diff)} lines differ (line counts {len(a) - 1} vs {len(b) - 1}); '
            f'{reorder} of them are reorderings within the line, {len(diff) - reorder} real differences'
        )
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv')
    src.add_argument('--seq')
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify-against')
    ap.add_argument('--protocol', choices=('legacy-v1', 'windowed-v2'), default='legacy-v1')
    ap.add_argument('--max-history-length', type=int, default=20)
    args = ap.parse_args()

    seqs = sequences_from_csv(args.csv) if args.csv else sequences_from_seq_file(args.seq)
    n_users = len(seqs)
    items = set()
    for _, s in seqs:
        items.update(s)
    print(
        f'users {n_users}, distinct items {len(items)}, max item id {max(items)}, actions {sum(len(s) for _, s in seqs)}'
    )
    train_u, valid_u, test_u = split_users(seqs)
    print(f'split: train {len(train_u)}, valid {len(valid_u)}, test {len(test_u)}')
    ladder = write_files(seqs, train_u, valid_u, test_u, args.out, args.protocol, args.max_history_length)
    print(f'ladder rows: {ladder}; files written to {args.out}')
    if args.verify_against:
        print(f'verification against {args.verify_against}:')
        ok = verify(args.out, args.verify_against)
        print('ALL FILES IDENTICAL' if ok else 'DIFFERENCES FOUND')
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
