import argparse
import ast
import re
from datetime import datetime
from pathlib import Path


EPOCH_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).* - Epoch \d+/\d+")


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


def parse_epoch_seconds(log_path: Path):
    times = []
    for line in log_path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.match(line)
        if match is None:
            continue
        times.append(datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f"))
    if len(times) < 2:
        return None
    gaps = [(times[idx] - times[idx - 1]).total_seconds() for idx in range(1, len(times))]
    return sum(gaps) / len(gaps)


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


def build_table(rows):
    headers = [
        "mode",
        "block",
        "stride",
        "bs",
        "lr",
        "head",
        "align",
        "loss_mode",
        "target",
        "shallow_layer",
        "item",
        "seed",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "epoch_sec",
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
        default="near,pre",
        help="Comma-separated align_item values to keep; use empty string to keep all",
    )
    args = parser.parse_args()
    keep_align_items = {
        item.strip() for item in args.align_items.split(",") if item.strip()
    }

    rows = []
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
        epoch_seconds = parse_epoch_seconds(log_path)
        align = f"{config.get('align_loss_type')}{fmt_float(config.get('mse_loss_weight'))}"
        rows.append(
            [
                fmt(config.get("train_mode")),
                fmt(config.get("block_items")),
                fmt(config.get("stride_items")),
                fmt(config.get("batch_size")),
                fmt_float(config.get("lr")),
                fmt(config.get("lm_head")),
                align,
                fmt(config.get("mse_loss_mode")),
                fmt(config.get("align_target")),
                (
                    fmt(config.get("shallow_layer"))
                    if config.get("align_target") == "shallow"
                    else "-"
                ),
                fmt(config.get("align_item")),
                fmt(config.get("seed")),
                f"{recalls.get('Recall@5', float('nan')):.6f}",
                f"{recalls.get('Recall@10', float('nan')):.6f}",
                f"{recalls.get('Recall@20', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@5', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@10', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@20', float('nan')):.6f}",
                "-" if epoch_seconds is None else f"{epoch_seconds:.2f}",
            ]
        )

    if not rows:
        print(f"No finished parallel logs found in {args.logs_dir}")
        return

    table = build_table(rows)
    Path(args.output).write_text(table + "\n")
    print(table)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
