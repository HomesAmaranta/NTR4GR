import argparse
import re
from pathlib import Path


TOTAL_RE = re.compile(r"Validation total loss:\s*([0-9.eE+-]+)")
CE_RE = re.compile(r"Validation CE loss:\s*([0-9.eE+-]+)")


def parse_validation_losses(log_path: Path):
    total_losses = []
    ce_losses = []
    for line in log_path.read_text(errors="ignore").splitlines():
        total_match = TOTAL_RE.search(line)
        if total_match is not None:
            total_losses.append(float(total_match.group(1)))
            continue
        ce_match = CE_RE.search(line)
        if ce_match is not None:
            ce_losses.append(float(ce_match.group(1)))

    n = min(len(total_losses), len(ce_losses))
    return total_losses[:n], ce_losses[:n]


def analyze_log(log_path: Path):
    total_losses, ce_losses = parse_validation_losses(log_path)
    if not total_losses:
        return None

    total_idx = min(range(len(total_losses)), key=total_losses.__getitem__)
    ce_idx = min(range(len(ce_losses)), key=ce_losses.__getitem__)
    return {
        "log": log_path.name,
        "epochs": len(total_losses),
        "total_min_epoch": total_idx + 1,
        "total_min": total_losses[total_idx],
        "ce_at_total_min": ce_losses[total_idx],
        "ce_min_epoch": ce_idx + 1,
        "ce_min": ce_losses[ce_idx],
        "total_at_ce_min": total_losses[ce_idx],
        "epoch_gap": ce_idx - total_idx,
        "total_gap": total_losses[ce_idx] - total_losses[total_idx],
        "ce_gap": ce_losses[total_idx] - ce_losses[ce_idx],
    }


def fmt_float(value):
    return f"{value:.6f}"


def build_table(rows):
    headers = [
        "log",
        "epochs",
        "total_min_epoch",
        "total_min",
        "ce_at_total_min",
        "ce_min_epoch",
        "ce_min",
        "total_at_ce_min",
        "epoch_gap(ce-total)",
        "total_gap",
        "ce_gap",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["log"],
                    str(row["epochs"]),
                    str(row["total_min_epoch"]),
                    fmt_float(row["total_min"]),
                    fmt_float(row["ce_at_total_min"]),
                    str(row["ce_min_epoch"]),
                    fmt_float(row["ce_min"]),
                    fmt_float(row["total_at_ce_min"]),
                    str(row["epoch_gap"]),
                    fmt_float(row["total_gap"]),
                    fmt_float(row["ce_gap"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs/0908_v3_shallow",
        help="Directory containing training log files.",
    )
    args = parser.parse_args()

    rows = []
    for log_path in sorted(Path(args.logs_dir).glob("*.log")):
        row = analyze_log(log_path)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No validation losses found in {args.logs_dir}")
        return

    print(build_table(rows))


if __name__ == "__main__":
    main()
