import argparse
import ast
import re
from pathlib import Path


LOG_NAME_RE = re.compile(
    r"^causal_tiger_(?P<dataset>.+)_(?P<loss_type>mse|cos)"
    r"(?P<weight>[0-9.]+)_(?P<mode>token|mean)"
    r"_shallow"
    r"(?:_(?P<align_item>pre|next))?"
    r"(?:_(?P<lm_head>emb|linear))?"
    r"_layer(?P<shallow_layer>[0-9]+)"
    r"(?:_seed(?P<seed>[0-9]+))?"
    r"\.log$"
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
        "align_target",
        "align_item",
        "lm_head",
        "shallow_layer",
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
    parser.add_argument("--output", type=str, default="./jepa_shallow_log_summary.md")
    parser.add_argument(
        "--include_unfinished",
        action="store_true",
        help="Include matched logs without final test metrics as empty metric rows.",
    )
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("*_shallow*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if match is None:
            continue
        metrics = parse_last_metrics(log_path)
        if metrics is None and not args.include_unfinished:
            continue
        recalls, ndcgs = metrics or ({}, {})
        rows.append(
            [
                log_path.name,
                match.group("dataset"),
                match.group("loss_type"),
                match.group("weight"),
                match.group("mode"),
                "shallow",
                match.group("align_item") or "next",
                match.group("lm_head") or "emb",
                match.group("shallow_layer"),
                match.group("seed") or "",
                format_metric(recalls.get("Recall@5")),
                format_metric(recalls.get("Recall@10")),
                format_metric(recalls.get("Recall@20")),
                format_metric(ndcgs.get("NDCG@5")),
                format_metric(ndcgs.get("NDCG@10")),
                format_metric(ndcgs.get("NDCG@20")),
            ]
        )

    table = build_table(rows)
    Path(args.output).write_text(table + "\n")
    print(table)
    print(f"\nsaved to {args.output}")


def format_metric(value):
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
