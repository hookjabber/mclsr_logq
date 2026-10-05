#!/usr/bin/env python
"""Rebuild the dataset files of the study from raw data, reproducibly.

Two inputs:
* ``--csv``  an Amazon ratings file (``user,item,rating,timestamp``, no header): the steps of
  ``notebooks/AmazonClothingDatasetUserSplit.ipynb`` -- iterative 5-core filtering on users and
  items, dense-rank ids (1..N in sorted order of the raw ids), rows sorted by (user, timestamp),
  one chronological sequence per user;
* ``--seq``  a text file with one line per user, ``user item item ...``, already 5-core with dense
  ids and in chronological order (how Beauty and CDs & Vinyl were prepared).

Then the shared protocol of the paper and of the study: users are shuffled with ``random.seed(42)``
and split 80/10/10; a training user keeps the last 20 items and contributes a prefix ladder (one
line per target position, ``train_mclsr.txt``) and the cropped sequence (``train_sasrec.txt``); a
validation / test user is split at ``int(0.8 * len)`` into history and targets.

``--verify-against DIR`` compares the produced files with existing ones line by line and, where
lines differ, says whether the difference is only a reordering within the line (e.g. a timestamp tie).

    python scripts/prepare_dataset.py --csv data/Clothing/data.csv --out /tmp/clothing_rebuilt \
        --verify-against data/Clothing
"""

import argparse
import os
import random

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


def write_files(seqs, train_u, valid_u, test_u, out):
    os.makedirs(out, exist_ok=True)
    names = ('train_mclsr', 'train_sasrec', 'valid_history', 'valid_target', 'test_history', 'test_target')
    fh = {n: open(os.path.join(out, n + '.txt'), 'w') for n in names}
    ladder = 0
    for uid, items in seqs:
        if uid in train_u:
            crop = items[-MAX_LEN_TRAIN:]
            fh['train_sasrec'].write(' '.join(map(str, [uid] + crop)) + '\n')
            for i in range(1, len(crop)):
                fh['train_mclsr'].write(' '.join(map(str, [uid] + crop[:i] + [crop[i]])) + '\n')
                ladder += 1
        else:
            assert len(items) >= 5
            si = int(len(items) * 0.8)
            assert 1 <= si < len(items)
            kind = 'valid' if uid in valid_u else 'test'
            fh[kind + '_history'].write(' '.join(map(str, [uid] + items[:si])) + '\n')
            fh[kind + '_target'].write(' '.join(map(str, [uid] + items[si:])) + '\n')
    for f in fh.values():
        f.close()
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
        diff = [(x, y) for x, y in zip(a, b) if x != y]
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
    ladder = write_files(seqs, train_u, valid_u, test_u, args.out)
    print(f'ladder rows: {ladder}; files written to {args.out}')
    if args.verify_against:
        print(f'verification against {args.verify_against}:')
        ok = verify(args.out, args.verify_against)
        print('ALL FILES IDENTICAL' if ok else 'DIFFERENCES FOUND')


if __name__ == '__main__':
    main()
