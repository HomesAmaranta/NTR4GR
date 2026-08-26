import argparse
import ast
import re
from pathlib import Path


LOG_NAME_RE = re.compile(
    r"^causal_tiger_(?P<dataset>.+)_(?P<loss_type>mse|cos)(?P<weight>[0-9.]+)_(?P<mode>token|mean)\.log$"
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
        "file",
        "dataset",
        "loss_type",
        "weight",
        "mode",
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue
        recalls, ndcgs = metrics
        rows.append(
            [
                log_path.name,
                match.group("dataset"),
                match.group("loss_type"),
                match.group("weight"),
                match.group("mode"),
                f"{recalls.get('Recall@5', float('nan')):.6f}",
                f"{recalls.get('Recall@10', float('nan')):.6f}",
                f"{recalls.get('Recall@20', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@5', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@10', float('nan')):.6f}",
                f"{ndcgs.get('NDCG@20', float('nan')):.6f}",
            ]
        )

    table = build_table(rows)
    if args.output:
        Path(args.output).write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
