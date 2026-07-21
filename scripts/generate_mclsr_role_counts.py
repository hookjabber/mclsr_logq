# use
# python scripts/generate_mclsr_role_counts.py --input ./data/Clothing/train_mclsr.txt \
#     --num_items 23033 --max_len 20 \
#     --target_output ./data/Clothing/item_target_counts.pkl \
#     --context_output ./data/Clothing/item_context_counts.pkl

import argparse
import hashlib
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

    with open(args.input, "rb") as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()

    # counts keep RAW ZEROS for absent ids (the numerical clamp lives in the
    # loss); the artifact is a self-describing dict so a loss can validate the
    # denominator/semantics instead of trusting hand-copied config numbers
    for role, counts, denominator, path in (
        ("target", target_counts, float(target_counts.sum()), args.target_output),
        ("context-inclusion", context_counts, float(lines), args.context_output),
    ):
        artifact = {
            "counts": counts,
            "denominator": denominator,
            "role": role,
            "max_len": args.max_len,
            "num_lines": lines,
            "source": args.input,
            "source_sha256": source_sha256,
        }
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"{role}: {lines} lines, sum={int(counts.sum())}, "
              f"zeros={int((counts == 0).sum())}, denominator={denominator:.0f} "
              f"-> {path}")


if __name__ == "__main__":
    main()
