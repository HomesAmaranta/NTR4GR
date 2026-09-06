import argparse
import ast
from pathlib import Path


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
        "train_mode",
        "lr",
        "head",
        "align",
        "loss_mode",
        "target",
        "item",
        "seed",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "log",
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
    parser.add_argument("--output", type=str, default="./non_parallel_align_summary.md")
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("causal_tiger_*_seed*.log")):
        if "parallel" in log_path.name:
            continue
        config = parse_config(log_path)
        if not config:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        recalls, ndcgs = metrics
        align = f"{config.get('align_loss_type')}{fmt_float(config.get('mse_loss_weight'))}"
        if align == "cos0":
            continue
        rows.append(
            [
                fmt(config.get("train_mode")),
                fmt_float(config.get("lr")),
                fmt(config.get("lm_head")),
                align,
                fmt(config.get("mse_loss_mode")),
                fmt(config.get("align_target")),
                fmt(config.get("align_item")),
                fmt(config.get("seed")),
                f"{recalls.get('Recall@5', float('nan')):.6f}",
                f"{recalls.get('Recall@10', float('nan')):.6f}",
                f"{recalls.get('Recall@20', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@5', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@10', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@20', float('nan')):.6f}",
                log_path.name,
            ]
        )

    if not rows:
        print(f"No finished non-parallel align logs found in {args.logs_dir}")
        return

    table = build_table(rows)
    Path(args.output).write_text(table + "\n")
    print(table)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
