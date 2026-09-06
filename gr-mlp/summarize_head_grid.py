import argparse
import ast
import re
from pathlib import Path


# Matches parallel head-grid logs, e.g.
#   causal_tiger_Beauty_parallel_b30_s10_bs256_lr1e-3_headmlp_seed2025.log
# bs/lr/head are optional (older logs default to 256 / 1e-3 / emb).
LOG_NAME_RE = re.compile(
    r"^causal_tiger_(?P<dataset>.+)_parallel"
    r"_b(?P<block>[0-9]+)_s(?P<stride>[0-9]+)"
    r"(?:_bs(?P<bs>[0-9]+))?"
    r"(?:_lr(?P<lr>[0-9.eE+-]+))?"
    r"(?:_head(?P<head>[A-Za-z-]+))"
    r"_seed(?P<seed>[0-9]+)\.log$"
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
        "block",
        "stride",
        "bs",
        "lr",
        "head",
        "seed",
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
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="./head_grid_summary.md")
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("*parallel*_seed*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        recalls, ndcgs = metrics
        rows.append(
            [
                match.group("block"),
                match.group("stride"),
                match.group("bs") or "256",
                match.group("lr") or "1e-3",
                match.group("head") or "emb",
                match.group("seed"),
                f"{recalls.get('Recall@5', float('nan')):.6f}",
                f"{recalls.get('Recall@10', float('nan')):.6f}",
                f"{recalls.get('Recall@20', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@5', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@10', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@20', float('nan')):.6f}",
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
