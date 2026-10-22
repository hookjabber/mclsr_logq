"""The dataset preparation CLI on a tiny synthetic input: the seed-42 user split, the cropped prefix
ladder, the int(0.8 * len) evaluation cut, and the self-check mode."""

import os
import random
import subprocess
import sys

import json
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import prepare_dataset  # noqa: E402


def _write_seq_file(path, num_users=50, seed=3):
    rng = random.Random(seed)
    seqs = []
    for uid in range(1, num_users + 1):
        length = rng.randint(5, 30)
        seqs.append((uid, [rng.randint(1, 60) for _ in range(length)]))
    with open(path, 'w') as f:
        for uid, items in seqs:
            f.write(' '.join(map(str, [uid] + items)) + '\n')
    return dict(seqs)


def test_split_ladder_and_eval_cut(tmp_path):
    src = str(tmp_path / 'seq.txt')
    seqs = _write_seq_file(src)
    out = str(tmp_path / 'out')
    read = prepare_dataset.sequences_from_seq_file(src)
    train_u, valid_u, test_u = prepare_dataset.split_users(read)
    assert len(train_u) == 40 and len(valid_u) == 5 and len(test_u) == 5
    assert not (train_u & valid_u or train_u & test_u or valid_u & test_u)
    # the split is a seed-42 shuffle of the user list, independent of this process's RNG state
    ids = list(range(1, 51))
    random.seed(42)
    random.shuffle(ids)
    assert train_u == set(ids[:40]) and valid_u == set(ids[40:45]) and test_u == set(ids[45:])

    ladder = prepare_dataset.write_files(read, train_u, valid_u, test_u, out)
    lines = {}
    for line in open(os.path.join(out, 'train_mclsr.txt')):
        p = [int(x) for x in line.split()]
        lines.setdefault(p[0], []).append(p[1:])
    assert ladder == sum(len(v) for v in lines.values()) and set(lines) == train_u
    for uid, rows in lines.items():
        crop = seqs[uid][-20:]
        assert rows == [crop[:i] + [crop[i]] for i in range(1, len(crop))]
    sas = {
        int(line.split()[0]): [int(x) for x in line.split()[1:]] for line in open(os.path.join(out, 'train_sasrec.txt'))
    }
    assert all(sas[u] == seqs[u][-20:] for u in train_u)
    for part, users in (('valid', valid_u), ('test', test_u)):
        hist = {
            int(line.split()[0]): [int(x) for x in line.split()[1:]]
            for line in open(os.path.join(out, f'{part}_history.txt'))
        }
        targ = {
            int(line.split()[0]): [int(x) for x in line.split()[1:]]
            for line in open(os.path.join(out, f'{part}_target.txt'))
        }
        assert set(hist) == users and set(targ) == users
        for u in users:
            cut = int(len(seqs[u]) * 0.8)
            assert hist[u] == seqs[u][:cut] and targ[u] == seqs[u][cut:]

    # --verify-against on its own output reports identity; a changed file is reported as a difference
    cli = [
        sys.executable,
        os.path.join(ROOT, 'scripts/prepare_dataset.py'),
        '--seq',
        src,
        '--out',
        str(tmp_path / 'again'),
        '--verify-against',
        out,
    ]
    res = subprocess.run(cli, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and 'ALL FILES IDENTICAL' in res.stdout, res.stdout + res.stderr
    with open(os.path.join(out, 'test_target.txt'), 'a') as f:
        f.write('999 1 2 3\n')
    res = subprocess.run(cli, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0
    assert 'DIFFERENCES FOUND' in res.stdout and 'test_target.txt' in res.stdout


def test_windowed_protocol_keeps_every_target_and_reserves_padding(tmp_path):
    seqs = [(0, list(range(25))), (8, [24, 0, 3, 7, 24]), (30, [0, 7, 9, 12, 24])]
    out = tmp_path / 'v2'
    rows = prepare_dataset.write_files(seqs, {0}, {8}, {30}, out, protocol='windowed-v2', max_history_length=20)
    ladder = [[int(value) for value in line.split()] for line in (out / 'train_mclsr.txt').read_text().splitlines()]
    assert rows == 24
    assert [row[-1] for row in ladder] == list(range(2, 26))
    assert max(len(row) - 2 for row in ladder) == 20
    assert ladder[-1] == [1] + list(range(5, 26))
    assert all(value > 0 for row in ladder for value in row)
    assert (out / 'valid_history.txt').read_text() == '2 25 1 4 8\n'
    assert (out / 'test_target.txt').read_text() == '3 25\n'
    manifest = json.loads((out / 'preprocessing.json').read_text())
    assert manifest['protocol'] == 'windowed-v2'
    assert manifest['max_history_length'] == 20
    assert manifest['item_id_mapping']['0'] == 1
    assert manifest['num_items'] == 25


def test_windowed_protocol_refuses_overwriting_existing_data(tmp_path):
    out = tmp_path / 'v2'
    out.mkdir()
    original = out / 'train_mclsr.txt'
    original.write_text('historical data\n')
    with pytest.raises(FileExistsError):
        prepare_dataset.write_files([(1, [0, 1, 2, 3, 4])], {1}, set(), set(), out, protocol='windowed-v2')
    assert original.read_text() == 'historical data\n'


def test_windowed_protocol_rejects_invalid_split_before_writing(tmp_path):
    with pytest.raises(ValueError, match='split'):
        prepare_dataset.write_files([(1, [1, 2, 3, 4, 5])], {1}, {1}, set(), tmp_path / 'bad', protocol='windowed-v2')
    assert not (tmp_path / 'bad').exists()
