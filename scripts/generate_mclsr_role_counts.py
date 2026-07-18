# use
# python scripts/generate_mclsr_role_counts.py --input ./data/Clothing/train_mclsr.txt \
#     --num_items 23033 --max_len 20 \
#     --target_output ./data/Clothing/item_target_counts.pkl \
#     --context_output ./data/Clothing/item_context_counts.pkl

import argparse
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Role-exact count tables for logQ from the mclsr ladder file. "
        "item_counts.pkl counts every event in full sequences — a proxy for both "
        "logQ uses. The exact proposals are: TARGET counts (how often an item is "
        "the last element of a ladder line = the L_P positive distribution) and "
        "CONTEXT-INCLUSION counts (in how many ladder line inputs an item occurs "
        "= the L_IC in-batch candidate distribution; unique per line, truncation "
        "applied exactly as in the train sampler).",
    )
    parser.add_argument("--input", required=True, help="Path to train_mclsr.txt")
    parser.add_argument("--num_items", type=int, required=True)
    parser.add_argument("--max_len", type=int, default=20,
                        help="max_sequence_length used in training (lines are cut "
                        "to the last max_len items before the target is split off)")
    parser.add_argument("--target_output", required=True)
    parser.add_argument("--context_output", required=True)
    args = parser.parse_args()

    size = args.num_items + 2
    target_counts = np.zeros(size, dtype=np.float32)
    context_counts = np.zeros(size, dtype=np.float32)

    lines = 0
    with open(args.input) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            item_ids = [int(i) for i in parts[1:]][-args.max_len:]
            target_counts[item_ids[-1]] += 1
            for item_id in set(item_ids[:-1]):
                context_counts[item_id] += 1
            lines += 1

    for name, counts, path in (
        ("target", target_counts, args.target_output),
        ("context-inclusion", context_counts, args.context_output),
    ):
        zeros = int((counts == 0).sum())
        counts = counts.copy()
        counts[counts == 0] = 1.0  # same zero-fill convention as item_counts.pkl
        with open(path, "wb") as f:
            pickle.dump(counts, f)
        print(f"{name}: {lines} lines, sum={int(counts.sum())}, "
              f"zero-filled={zeros} -> {path}")


if __name__ == "__main__":
    main()
