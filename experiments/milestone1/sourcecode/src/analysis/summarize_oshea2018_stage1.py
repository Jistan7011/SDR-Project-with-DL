from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_eval(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(data: dict[str, object], key: str) -> float:
    value = data.get(key, 0.0)
    return float(value) if value is not None else 0.0


def class_recall(data: dict[str, object], name: str) -> float:
    report = data.get("classification_report", {})
    if isinstance(report, dict) and name in report and isinstance(report[name], dict):
        return float(report[name].get("recall", 0.0))
    recalls = data.get("class_recall", {})
    if isinstance(recalls, dict):
        return float(recalls.get(name, 0.0))
    return 0.0


def confusion_matrix(data: dict[str, object]) -> list[list[int]]:
    value = data.get("confusion_matrix", [])
    if not isinstance(value, list):
        return []
    matrix: list[list[int]] = []
    for row in value:
        if isinstance(row, list):
            matrix.append([int(item) for item in row])
    return matrix


def worst_recall(data: dict[str, object]) -> float:
    explicit = data.get("worst_recall", None)
    if explicit is not None:
        return float(explicit)
    recalls = [class_recall(data, name) for name in ("BASK", "BFSK", "BPSK")]
    return min(recalls) if recalls else 0.0


def bpsk_all_to_bask(data: dict[str, object]) -> bool:
    matrix = confusion_matrix(data)
    if len(matrix) < 3 or len(matrix[2]) < 3:
        return False
    bpsk_total = sum(matrix[2])
    return bpsk_total > 0 and matrix[2][0] == bpsk_total


def summarize(results_root: str, prefix: str, output: str, min_bpsk_recall: float) -> dict[str, object]:
    root = Path(results_root)
    rows: list[dict[str, object]] = []
    for eval_path in sorted(root.glob(f"{prefix}*/logs/eval_test.json")):
        data = load_eval(eval_path)
        if data is None:
            continue
        model = eval_path.parents[1].name
        row = {
            "model": model,
            "eval_path": str(eval_path),
            "accuracy": metric(data, "accuracy"),
            "macro_f1": metric(data, "macro_f1"),
            "worst_recall": worst_recall(data),
            "bask_recall": class_recall(data, "BASK"),
            "bfsk_recall": class_recall(data, "BFSK"),
            "bpsk_recall": class_recall(data, "BPSK"),
            "bfsk_bpsk_to_bask_rate": metric(data, "bfsk_bpsk_to_bask_rate"),
            "bpsk_all_to_bask": bpsk_all_to_bask(data),
        }
        row["stage2_pass"] = bool(
            row["bpsk_recall"] >= min_bpsk_recall
            and row["worst_recall"] > 0.0
            and not row["bpsk_all_to_bask"]
        )
        rows.append(row)
    rows.sort(key=lambda row: (bool(row["stage2_pass"]), float(row["bpsk_recall"]), float(row["accuracy"])), reverse=True)
    summary = {
        "results_root": str(root),
        "prefix": prefix,
        "min_bpsk_recall": min_bpsk_recall,
        "stage2_allowed": any(bool(row["stage2_pass"]) for row in rows),
        "rows": rows,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out.with_suffix(".md"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Oshea2018 Fixed Stage1 Summary",
        "",
        f"- stage2_allowed: `{summary['stage2_allowed']}`",
        f"- min_bpsk_recall: `{summary['min_bpsk_recall']}`",
        "",
        "| model | acc | macro F1 | worst | BASK | BFSK | BPSK | absorption | BPSK all->BASK | Stage2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {model} | {accuracy:.4f} | {macro_f1:.4f} | {worst_recall:.4f} | {bask_recall:.4f} | {bfsk_recall:.4f} | {bpsk_recall:.4f} | {bfsk_bpsk_to_bask_rate:.4f} | {bpsk_all_to_bask} | {stage2} |".format(
                model=row["model"],
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                worst_recall=row["worst_recall"],
                bask_recall=row["bask_recall"],
                bfsk_recall=row["bfsk_recall"],
                bpsk_recall=row["bpsk_recall"],
                bfsk_bpsk_to_bask_rate=row["bfsk_bpsk_to_bask_rate"],
                bpsk_all_to_bask="YES" if row["bpsk_all_to_bask"] else "NO",
                stage2="PASS" if row["stage2_pass"] else "HOLD",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="../results")
    parser.add_argument("--prefix", default="fixed_quick_")
    parser.add_argument("--output", default="../results/fixed_stage1_summary.json")
    parser.add_argument("--min-bpsk-recall", type=float, default=0.30)
    args = parser.parse_args()
    summarize(args.results_root, args.prefix, args.output, args.min_bpsk_recall)


if __name__ == "__main__":
    main()
