# use
# python scripts/generate_train_presence.py --mode item --input ./data/Clothing/train_sasrec.txt --output ./data/Clothing/train_presence_items.pkl --num_entities 23033
# python scripts/generate_train_presence.py --mode user --input ./data/Clothing/train_mclsr.txt --output ./data/Clothing/train_presence_users.pkl --num_entities 39387

import argparse
import os
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate a boolean train-presence mask (True = entity occurs in the "
        "train split). Unlike the *_counts.pkl tables, absent entities are NOT filled "
        "with 1.0 — this is the mask the matched full-catalog contrastive loss uses to "
        "exclude non-train entities without touching real train singletons.",
    )
    parser.add_argument("--mode", choices=["user", "item"], required=True,
                        help="user: entity id is the first column; item: ids are columns 1+")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the train split (train_mclsr.txt for users, train_sasrec.txt for items)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_entities", type=int, required=True,
                        help="Number of entities from dataset meta; array size is num_entities + 2 "
                        "(padding at 0, mask at num_entities + 1) — same layout as the counts tables")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    array_size = args.num_entities + 2
    presence = np.zeros(array_size, dtype=bool)

    with open(args.input) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ids = [int(parts[0])] if args.mode == "user" else [int(i) for i in parts[1:]]
            for entity_id in ids:
                if entity_id >= array_size:
                    raise ValueError(
                        f"{args.mode}_id {entity_id} exceeds array size {array_size}. Check --num_entities."
                    )
                presence[entity_id] = True

    with open(args.output, "wb") as f:
        pickle.dump(presence, f)

    print(f"Saved {args.mode} train presence to {args.output}: "
          f"{int(presence.sum())}/{array_size} present")


if __name__ == "__main__":
    main()
