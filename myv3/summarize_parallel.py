import argparse
import ast
import re
from pathlib import Path


FINAL_EPOCH_RE = re.compile(r"Final Epoch:\s*(?P<epoch>\d+)")
METRIC_KEYS = [
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
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


def parse_last_metrics(log_path: Path):
    last_recalls = None
    last_ndcgs = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Final Test Recalls:" in line:
            last_recalls = ast.literal_eval(line.split("Final Test Recalls:", 1)[1].strip())
        elif "Final Test NDCGs:" in line:
            last_ndcgs = ast.literal_eval(line.split("Final Test NDCGs:", 1)[1].strip())
    if last_recalls is None or last_ndcgs is None:
        return None
    return last_recalls, last_ndcgs


def parse_converged_epoch(log_path: Path):
    converged_epoch = None
    for line in log_path.read_text(errors="ignore").splitlines():
        match = FINAL_EPOCH_RE.search(line)
        if match is not None:
            converged_epoch = int(match.group("epoch"))
    return converged_epoch


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


def hidden_layer_sort_key(value):
    try:
        layer = int(value)
    except (TypeError, ValueError):
        return 10**9
    if layer < 0:
        return abs(layer) - 1
    return 10**6 + layer


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def entry_sort_key(entry):
    return (
        hidden_layer_sort_key(entry["hidden_layer_raw"]),
        entry["lr_raw"],
        entry["target"],
        entry["align_mode"],
        entry["item"],
        entry["k"],
        entry["loss_mode"],
        entry["seed"],
        entry["log_name"],
    )


def entry_to_row(entry):
    return [
        entry["align"],
        entry["loss_mode"],
        entry["target"],
        entry["align_mode"],
        entry["shallow_layer"],
        entry["hidden_layer"],
        entry["lr"],
        entry["item"],
        entry["k"],
        entry["seed"],
        f"{entry['Recall@5']:.6f}",
        f"{entry['Recall@10']:.6f}",
        f"{entry['Recall@20']:.6f}",
        f"{entry['NDCG@5']:.6f}",
        f"{entry['NDCG@10']:.6f}",
        f"{entry['NDCG@20']:.6f}",
        fmt_float(entry["conv_epoch"]),
    ]


def aggregate_entries(entries):
    groups = {}
    for entry in entries:
        group_key = (
            entry["align"],
            entry["loss_mode"],
            entry["target"],
            entry["align_mode"],
            entry["shallow_layer"],
            entry["hidden_layer"],
            entry["lr"],
            entry["item"],
            entry["k"],
        )
        groups.setdefault(group_key, []).append(entry)

    aggregated = []
    for group in groups.values():
        base = dict(group[0])
        base["seed"] = f"mean({len(group)})"
        base["log_name"] = ""
        for key in METRIC_KEYS:
            base[key] = mean([entry[key] for entry in group])
        base["conv_epoch"] = mean([entry["conv_epoch"] for entry in group])
        aggregated.append(base)
    return aggregated


def build_table(rows):
    headers = [
        "align",
        "loss_mode",
        "target",
        "align_mode",
        "shallow_layer",
        "hidden_layer",
        "lr",
        "item",
        "k",
        "seed",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "conv_epoch",
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
    parser.add_argument("--logs_dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="./parallel_0904_all_summary.md")
    parser.add_argument(
        "--align_items",
        type=str,
        default="",
        help="Comma-separated align_item values to keep; use empty string to keep all",
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Average rows with the same config across seeds",
    )
    args = parser.parse_args()
    keep_align_items = {
        item.strip() for item in args.align_items.split(",") if item.strip()
    }

    entries = []
    for log_path in sorted(Path(args.logs_dir).glob("*.log")):
        config = parse_config(log_path)
        if not config or config.get("train_mode") != "parallel":
            continue
        if keep_align_items and config.get("align_item") not in keep_align_items:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        recalls, ndcgs = metrics
        converged_epoch = parse_converged_epoch(log_path)
        align = f"{config.get('align_loss_type')}{fmt_float(config.get('mse_loss_weight'))}"
        hidden_layer = config.get("hidden_layer", -1)
        lr = config.get("lr")
        entries.append(
            {
                "align": align,
                "loss_mode": fmt(config.get("mse_loss_mode")),
                "target": fmt(config.get("align_target")),
                "align_mode": fmt(config.get("align_mode", "add")),
                "shallow_layer": (
                    fmt(config.get("shallow_layer"))
                    if config.get("align_target") == "shallow"
                    else "-"
                ),
                "hidden_layer": fmt(config.get("hidden_layer", -1)),
                "hidden_layer_raw": hidden_layer,
                "lr": fmt_float(lr),
                "lr_raw": float(lr) if lr is not None else float("inf"),
                "item": fmt(config.get("align_item")),
                "k": fmt(config.get("align_current_k")),
                "seed": fmt(config.get("seed")),
                "Recall@5": recalls.get("Recall@5", float("nan")),
                "Recall@10": recalls.get("Recall@10", float("nan")),
                "Recall@20": recalls.get("Recall@20", float("nan")),
                "NDCG@5": ndcgs.get("NDCG@5", float("nan")),
                "NDCG@10": ndcgs.get("NDCG@10", float("nan")),
                "NDCG@20": ndcgs.get("NDCG@20", float("nan")),
                "conv_epoch": converged_epoch,
                "log_name": log_path.name,
            }
        )

    if not entries:
        print(f"No finished parallel logs found in {args.logs_dir}")
        return

    if args.mean:
        entries = aggregate_entries(entries)
    rows = [entry_to_row(entry) for entry in sorted(entries, key=entry_sort_key)]
    table = build_table(rows)
    # Path(args.output).write_text(table + "\n")
    print(table)
    # print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
