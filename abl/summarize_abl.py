import argparse
import ast
import re
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)")
FINAL_TEST_MARKER = "Final Test CE Loss:"
METRIC_KEYS = [
    "test_ce",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "SID1_HR@5",
    "SID2_HR@5",
    "SID3_HR@5",
    "SID4_HR@5",
    "SID1_CondHR@5",
    "SID2_CondHR@5",
    "SID3_CondHR@5",
    "SID4_CondHR@5",
    "SID1_HR@10",
    "SID2_HR@10",
    "SID3_HR@10",
    "SID4_HR@10",
    "SID1_CondHR@10",
    "SID2_CondHR@10",
    "SID3_CondHR@10",
    "SID4_CondHR@10",
]


def parse_config(log_path: Path):
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Configuration:" not in line:
            continue
        raw = line.split("Configuration:", 1)[1].strip()
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    return None


def parse_final_metrics(log_path: Path):
    last_line = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if FINAL_TEST_MARKER in line:
            last_line = line
    if last_line is None:
        return None

    try:
        tail = last_line.split(FINAL_TEST_MARKER, 1)[1].strip()
        ce_raw, tail = tail.split(", Recalls:", 1)
        recalls_raw, tail = tail.split(", NDCGs:", 1)
        if ", SID_HR:" in tail:
            ndcgs_raw, sid_hr_raw = tail.split(", SID_HR:", 1)
            sid_hr = ast.literal_eval(sid_hr_raw.strip())
        else:
            ndcgs_raw = tail
            sid_hr = {}
        return {
            "test_ce": float(ce_raw.strip()),
            "recalls": ast.literal_eval(recalls_raw.strip()),
            "ndcgs": ast.literal_eval(ndcgs_raw.strip()),
            "sid_hr": sid_hr,
        }
    except (ValueError, SyntaxError):
        return None


def parse_epochs(log_path: Path):
    final_epoch = None
    best_epoch = None
    previous_epoch = None
    for line in log_path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match is not None:
            previous_epoch = int(match.group("epoch"))
            final_epoch = previous_epoch
        elif "Saved best model" in line and previous_epoch is not None:
            best_epoch = previous_epoch
    return final_epoch, best_epoch


def infer_target(config):
    path = str(config.get("item_emb_path", ""))
    if "quantized" in path:
        return "quantized"
    if "encoder_latent" in path or "latent" in path:
        return "latent"
    if "codebook" in path:
        return "codebook"
    if path.endswith("item_emb.parquet"):
        return "item"
    return "-"


def fmt(value):
    if value is None:
        return "-"
    return str(value)


def fmt_float(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def entry_sort_key(entry):
    return (
        entry["target"],
        entry["source_offset_raw"],
        entry["max_source_offset_raw"],
        entry["lr_raw"],
        entry["hidden_dim_raw"],
        entry["mlp_dim_raw"],
        entry["beam_size_raw"],
        entry["seed"],
        entry["log_name"],
    )


def entry_to_row(entry):
    return [
        entry["target"],
        entry["source_offset"],
        entry["max_source_offset"],
        entry["hidden_dim"],
        entry["mlp_dim"],
        entry["dropout"],
        entry["lr"],
        entry["beam_size"],
        entry["seed"],
        fmt_float(entry["test_ce"]),
        f"{entry['Recall@5']:.6f}",
        f"{entry['Recall@10']:.6f}",
        f"{entry['Recall@20']:.6f}",
        f"{entry['NDCG@5']:.6f}",
        f"{entry['NDCG@10']:.6f}",
        f"{entry['NDCG@20']:.6f}",
        f"{entry['SID1_HR@5']:.6f}",
        f"{entry['SID2_HR@5']:.6f}",
        f"{entry['SID3_HR@5']:.6f}",
        f"{entry['SID4_HR@5']:.6f}",
        f"{entry['SID1_HR@10']:.6f}",
        f"{entry['SID2_HR@10']:.6f}",
        f"{entry['SID3_HR@10']:.6f}",
        f"{entry['SID4_HR@10']:.6f}",
        fmt_float(entry["best_epoch"]),
        fmt_float(entry["final_epoch"]),
    ]


def aggregate_entries(entries):
    groups = {}
    for entry in entries:
        group_key = (
            entry["target"],
            entry["source_offset"],
            entry["max_source_offset"],
            entry["hidden_dim"],
            entry["mlp_dim"],
            entry["dropout"],
            entry["lr"],
            entry["beam_size"],
        )
        groups.setdefault(group_key, []).append(entry)

    aggregated = []
    for group in groups.values():
        base = dict(group[0])
        base["seed"] = f"mean({len(group)})"
        base["log_name"] = ""
        for key in METRIC_KEYS:
            base[key] = mean([entry[key] for entry in group])
        base["best_epoch"] = mean([entry["best_epoch"] for entry in group])
        base["final_epoch"] = mean([entry["final_epoch"] for entry in group])
        aggregated.append(base)
    return aggregated


def build_table(rows):
    headers = [
        "target",
        "source_offset",
        "max_source_offset",
        "hidden_dim",
        "mlp_dim",
        "dropout",
        "lr",
        "beam",
        "seed",
        "test_ce",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "SID1_HR@5",
        "SID2_HR@5",
        "SID3_HR@5",
        "SID4_HR@5",
        "SID1_HR@10",
        "SID2_HR@10",
        "SID3_HR@10",
        "SID4_HR@10",
        "best_epoch",
        "final_epoch",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="./0916_abl_offset_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated target values to keep; use empty string to keep all",
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Average rows with the same config across seeds",
    )
    args = parser.parse_args()
    keep_targets = {
        target.strip() for target in args.targets.split(",") if target.strip()
    }

    entries = []
    for log_path in sorted(Path(args.logs_dir).glob("*.log")):
        config = parse_config(log_path)
        if not config:
            continue
        target = infer_target(config)
        if keep_targets and target not in keep_targets:
            continue
        metrics = parse_final_metrics(log_path)
        if metrics is None:
            continue
        final_epoch, best_epoch = parse_epochs(log_path)
        recalls = metrics["recalls"]
        ndcgs = metrics["ndcgs"]
        sid_hr = metrics["sid_hr"]
        source_offset = config.get("source_offset")
        max_source_offset = config.get("max_source_offset")
        lr = config.get("lr")
        hidden_dim = config.get("hidden_dim")
        mlp_dim = config.get("mlp_dim")
        beam_size = config.get("beam_size")
        entry = {
            "target": target,
            "source_offset": fmt(source_offset),
            "source_offset_raw": (
                int(source_offset) if source_offset is not None else 10**9
            ),
            "max_source_offset": fmt(max_source_offset),
            "max_source_offset_raw": (
                int(max_source_offset) if max_source_offset is not None else 10**9
            ),
            "hidden_dim": fmt(hidden_dim),
            "hidden_dim_raw": int(hidden_dim) if hidden_dim is not None else 10**9,
            "mlp_dim": fmt(mlp_dim),
            "mlp_dim_raw": int(mlp_dim) if mlp_dim is not None else 10**9,
            "dropout": fmt_float(config.get("dropout")),
            "lr": fmt_float(lr),
            "lr_raw": float(lr) if lr is not None else float("inf"),
            "beam_size": fmt(beam_size),
            "beam_size_raw": int(beam_size) if beam_size is not None else 10**9,
            "seed": fmt(config.get("seed")),
            "test_ce": metrics["test_ce"],
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "log_name": log_path.name,
        }
        for key in METRIC_KEYS:
            if key == "test_ce":
                continue
            if key.startswith("Recall"):
                entry[key] = recalls.get(key, float("nan"))
            elif key.startswith("NDCG"):
                entry[key] = ndcgs.get(key, float("nan"))
            else:
                entry[key] = sid_hr.get(key, float("nan"))
        entries.append(entry)

    if not entries:
        print(f"No finished abl logs found in {args.logs_dir}")
        return

    if args.mean:
        entries = aggregate_entries(entries)
    rows = [entry_to_row(entry) for entry in sorted(entries, key=entry_sort_key)]
    table = build_table(rows)
    if args.output:
        Path(args.output).write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
