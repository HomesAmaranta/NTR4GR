import ast
import re
import sys


ORDER = [
    ("Recall@5", "recall"),
    ("Recall@10", "recall"),
    ("Recall@20", "recall"),
    ("NDCG@5", "ndcg"),
    ("NDCG@10", "ndcg"),
    ("NDCG@20", "ndcg"),
]
DIGITS = 4

DICT_RE = re.compile(r"\{[^{}]*\}")


def parse_dict(text: str) -> dict:
    m = DICT_RE.search(text)
    if not m:
        return {}
    try:
        return ast.literal_eval(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def format_metrics(text: str) -> str:
    recalls, ndcgs = {}, {}
    for line in text.splitlines():
        if "Recall" in line and "Recall@" in line:
            recalls.update(parse_dict(line))
        if "NDCG@" in line:
            ndcgs.update(parse_dict(line))
    out = []
    for key, kind in ORDER:
        src = recalls if kind == "recall" else ndcgs
        val = src.get(key)
        out.append("" if val is None else f"{val:.{DIGITS}f}")
    row = "| " + " | ".join(out) + " |"
    sep = "| " + " | ".join(["---"] * len(ORDER)) + " |"
    return "\n".join([row, sep])


def main():
    print(
        "Paste the log text (the Recall / NDCG lines), "
        "then an empty line to output. Ctrl-C / Ctrl-D to quit."
    )
    buf = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            if buf:
                print(format_metrics("\n".join(buf)))
            print()
            break
        if line.strip() == "":
            if buf:
                print(format_metrics("\n".join(buf)))
                buf = []
            continue
        buf.append(line)


if __name__ == "__main__":
    main()
