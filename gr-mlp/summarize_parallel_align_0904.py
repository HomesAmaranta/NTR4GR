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
    r"_seed(?P<seed>[0-9]+)_0904\.log$"
)


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
    parser.add_argument("--output", type=str, default="./parallel_align_0904_summary.md")
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("*_0904.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        recalls, ndcgs = metrics
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
        print(f"No finished 0904 parallel align logs found in {args.logs_dir}")
        return

    table = build_table(rows)
    Path(args.output).write_text(table + "\n")
    print(table)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
