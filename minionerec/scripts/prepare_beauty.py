#!/usr/bin/env python3
import argparse
import ast
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SID_PREFIXES = ("a", "b", "c", "d")
CSV_FIELDS = (
    "user_id",
    "history_item_title",
    "item_title",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    ntr4gr_dir = project_dir.parent
    parser = argparse.ArgumentParser(
        description="Convert NTR4GR Beauty data to MiniOneRec's four-level SID format."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ntr4gr_dir / "data" / "Beauty",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "Beauty",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=20,
        help="Keep the most recent N items, matching NTR4GR's default.",
    )
    parser.add_argument(
        "--no-expand-train",
        action="store_true",
        help="Keep one row per user instead of NTR4GR's train-prefix expansion.",
    )
    return parser.parse_args()


def load_item_mapping(path):
    mapping = np.load(path, allow_pickle=True).item()
    if not isinstance(mapping, dict):
        raise TypeError(f"{path} must contain an ASIN-to-ItemID dictionary")

    normalized = {str(asin): int(item_id) for asin, item_id in mapping.items()}
    expected = set(range(1, len(normalized) + 1))
    actual = set(normalized.values())
    if actual != expected:
        raise ValueError("Item IDs must be consecutive and 1-based")
    return normalized


def load_indices(path, item_count):
    codes = np.load(path, allow_pickle=False)
    if codes.shape != (item_count, len(SID_PREFIXES)):
        raise ValueError(
            f"Expected SID shape {(item_count, len(SID_PREFIXES))}, got {codes.shape}"
        )
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError(f"SID codes must be integers, got {codes.dtype}")
    if np.any(codes < 0):
        raise ValueError("SID codes must be non-negative")
    if len(np.unique(codes, axis=0)) != item_count:
        raise ValueError("Four-level SIDs must uniquely identify every item")

    indices = {}
    combined = {}
    for row_index, code in enumerate(codes):
        item_id = str(row_index + 1)
        tokens = [
            f"<{prefix}_{int(value)}>"
            for prefix, value in zip(SID_PREFIXES, code)
        ]
        indices[item_id] = tokens
        combined[int(item_id)] = "".join(tokens)
    return indices, combined


def load_item_features(path, asin_to_item_id):
    features = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                metadata = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid metadata at line {line_number}") from exc

            asin = str(metadata.get("asin", ""))
            item_id = asin_to_item_id.get(asin)
            if item_id is None:
                continue

            title = str(metadata.get("title") or "").strip() or f"Item_{item_id}"
            features[str(item_id)] = {
                "title": title,
                "description": metadata.get("description") or title,
                "brand": metadata.get("brand"),
                "categories": metadata.get("categories"),
                "price": metadata.get("price"),
                "salesRank": metadata.get("salesRank"),
                "asin": asin,
            }

    missing = sorted(set(range(1, len(asin_to_item_id) + 1)) - {int(x) for x in features})
    if missing:
        raise ValueError(f"Missing metadata for ItemIDs: {missing[:20]}")
    return features


def clean_info_text(value):
    return " ".join(str(value).replace("\t", " ").splitlines()).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
    os.replace(temporary, path)


def write_info(path, item_features, combined_sids):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for item_id in range(1, len(combined_sids) + 1):
            title = clean_info_text(item_features[str(item_id)]["title"])
            stream.write(f"{combined_sids[item_id]}\t{title}\t{item_id}\n")
    os.replace(temporary, path)


def make_row(user_id, history, target, item_features, combined_sids, max_history):
    history = [int(item_id) for item_id in history][-max_history:]
    target = int(target)
    history_sids = [combined_sids[item_id] for item_id in history]
    return {
        "user_id": int(user_id),
        "history_item_title": [
            item_features[str(item_id)]["title"] for item_id in history
        ],
        "item_title": item_features[str(target)]["title"],
        "history_item_id": history,
        "item_id": target,
        "history_item_sid": history_sids,
        "item_sid": combined_sids[target],
    }


def convert_split(
    source_path,
    output_path,
    item_features,
    combined_sids,
    max_history,
    expand_train,
):
    table = pq.read_table(source_path, columns=["user", "history", "target"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    row_count = 0

    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in table.to_pylist():
            history = [int(item_id) for item_id in record["history"]]
            target = int(record["target"])
            if expand_train:
                sequence = history + [target]
                samples = (
                    (sequence[:target_index], sequence[target_index])
                    for target_index in range(1, len(sequence))
                )
            else:
                samples = ((history, target),)

            for sample_history, sample_target in samples:
                writer.writerow(
                    make_row(
                        record["user"],
                        sample_history,
                        sample_target,
                        item_features,
                        combined_sids,
                        max_history,
                    )
                )
                row_count += 1

    os.replace(temporary, output_path)
    return row_count


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_history <= 0:
        raise ValueError("--max-history must be positive")

    asin_to_item_id = load_item_mapping(source_dir / "item_mapping.npy")
    indices, combined_sids = load_indices(
        source_dir / "Beauty_t5_rqvae.npy", len(asin_to_item_id)
    )
    item_features = load_item_features(
        source_dir / "meta_Beauty.json.gz", asin_to_item_id
    )

    write_json(output_dir / "index" / "Beauty.index.json", indices)
    write_json(output_dir / "index" / "Beauty.item.json", item_features)
    write_info(output_dir / "info" / "Beauty.txt", item_features, combined_sids)

    split_counts = {}
    for split in ("train", "valid", "test"):
        split_counts[split] = convert_split(
            source_dir / f"{split}.parquet",
            output_dir / split / "Beauty.csv",
            item_features,
            combined_sids,
            args.max_history,
            expand_train=split == "train" and not args.no_expand_train,
        )

    summary = {
        "source_dir": str(source_dir),
        "item_count": len(asin_to_item_id),
        "sid_levels": len(SID_PREFIXES),
        "max_history": args.max_history,
        "train_prefix_expansion": not args.no_expand_train,
        "split_rows": split_counts,
    }
    write_json(output_dir / "conversion_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
