import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def to_array(embedding):
    arr = np.asarray(embedding)
    if arr.dtype == object:
        arr = np.stack(embedding)
    return arr.astype(np.float32)


def load_item_embeddings(item_emb_path):
    data = pd.read_parquet(item_emb_path)
    return {
        int(row.ItemID): to_array(row.embedding)
        for row in data.itertuples(index=False)
    }


def load_full_sequences(dataset_dir, splits):
    sequences_by_user = {}
    for split in splits:
        split_path = dataset_dir / f"{split}.parquet"
        if not split_path.exists():
            continue
        data = pd.read_parquet(split_path)
        for row in data.itertuples(index=False):
            sequence = [int(item) for item in list(row.history)] + [int(row.target)]
            user = int(row.user)
            previous = sequences_by_user.get(user)
            if previous is None or len(sequence) > len(previous):
                sequences_by_user[user] = sequence
    return sequences_by_user


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../data/Beauty")
    parser.add_argument(
        "--item_emb_path",
        type=str,
        default="../data/Beauty/item_emb_rqvae_quantized_latent.parquet",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated parquet splits to merge by user",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=1,
        help="Item distance to compare; 1 means adjacent, 2 means a-c in a,b,c",
    )
    args = parser.parse_args()
    if args.gap <= 0:
        raise ValueError("--gap must be positive")

    dataset_dir = Path(args.dataset_dir)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    sequences_by_user = load_full_sequences(dataset_dir, splits)
    item_embeddings = load_item_embeddings(args.item_emb_path)

    sequence_lengths = []
    cosine_sum = 0.0
    pair_count = 0
    missing_pair_count = 0

    for sequence in sequences_by_user.values():
        sequence_lengths.append(len(sequence))
        for left, right in zip(sequence[:-args.gap], sequence[args.gap:]):
            left_emb = item_embeddings.get(int(left))
            right_emb = item_embeddings.get(int(right))
            if left_emb is None or right_emb is None:
                missing_pair_count += 1
                continue
            cos = cosine(left_emb, right_emb)
            if cos is None:
                missing_pair_count += 1
                continue
            cosine_sum += cos
            pair_count += 1

    sequence_count = len(sequence_lengths)
    avg_length = sum(sequence_lengths) / sequence_count if sequence_count else 0.0
    avg_cosine = cosine_sum / pair_count if pair_count else float("nan")

    print(f"gap: {args.gap}")
    print(f"sequence_count: {sequence_count}")
    print(f"average_sequence_length: {avg_length:.6f}")
    print(f"pair_count: {pair_count}")
    print(f"missing_pair_count: {missing_pair_count}")
    print(f"average_gap{args.gap}_quantized_cosine: {avg_cosine:.6f}")


if __name__ == "__main__":
    main()
