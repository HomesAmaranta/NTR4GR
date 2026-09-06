import argparse
import ast
import re
from pathlib import Path


LOG_NAME_RE = re.compile(
    r"^causal_tiger_(?P<dataset>.+)_(?P<mode>parallel)"
    r"_b(?P<block>[0-9]+)_s(?P<stride>[0-9]+)"
    r"_bs(?P<bs>[0-9]+)"
    r"_lr(?P<lr>[0-9.eE+-]+)"
    r"_head(?P<head>[A-Za-z-]+)"
    r"_(?P<loss_type>cos|mse)(?P<loss_weight>[0-9.]+)"
    r"_(?P<loss_mode>[A-Za-z-]+)"
    r"_(?P<align_target>[A-Za-z-]+)"
    r"_(?P<align_item>[A-Za-z-]+)"
    r"_seed(?P<seed>[0-9]+)\.log$"
)


def parse_last_metrics(log_path: Path):
    final_epoch = None
    last_recalls = None
    last_ndcgs = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Final Epoch:" in line:
            final_epoch = line.split("Final Epoch:", 1)[1].strip()
        elif "Final Test Recalls:" in line:
            last_recalls = ast.literal_eval(line.split("Final Test Recalls:", 1)[1].strip())
        elif "Final Test NDCGs:" in line:
            last_ndcgs = ast.literal_eval(line.split("Final Test NDCGs:", 1)[1].strip())
    if last_recalls is None or last_ndcgs is None:
        return None
    return final_epoch or "", last_recalls, last_ndcgs


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
        "item",
        "seed",
        "epoch",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="Directory containing parallel align logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to <logs_dir>/summary.md.",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir
    output = args.output or logs_dir / "summary.md"
    rows = []
    for log_path in sorted(logs_dir.glob("*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        final_epoch, recalls, ndcgs = metrics
        align = f"{match.group('loss_type')}{match.group('loss_weight')}"
        rows.append(
            [
                match.group("mode"),
                match.group("block"),
                match.group("stride"),
                match.group("bs"),
                match.group("lr"),
                match.group("head"),
                align,
                match.group("loss_mode"),
                match.group("align_target"),
                match.group("align_item"),
                match.group("seed"),
                final_epoch,
                f"{recalls.get('Recall@5', float('nan')):.6f}",
                f"{recalls.get('Recall@10', float('nan')):.6f}",
                f"{recalls.get('Recall@20', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@5', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@10', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@20', float('nan')):.6f}",
            ]
        )

    if not rows:
        print(f"No finished parallel align logs found in {logs_dir}")
        return

    table = build_table(rows)
    output.write_text(table + "\n")
    print(table)
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()
