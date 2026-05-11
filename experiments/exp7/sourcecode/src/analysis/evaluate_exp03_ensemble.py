from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def load_checkpoint_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
        input_channels=int(model_cfg["input_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def predict_probabilities(checkpoint_path: Path, data_root: Path, split: str, batch_size: int) -> tuple[list[Path], np.ndarray, np.ndarray]:
    model, cfg = load_checkpoint_model(checkpoint_path)
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    dataset = IQDataset(data_root, split, feature_mode=feature_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return dataset.files, np.concatenate(labels), np.concatenate(probs)


def group_probabilities(members: dict[str, list[Path]], data_root: Path, split: str, batch_size: int) -> tuple[list[Path], np.ndarray, dict[str, np.ndarray]]:
    files_ref: list[Path] | None = None
    labels_ref: np.ndarray | None = None
    grouped: dict[str, np.ndarray] = {}
    for group, checkpoints in members.items():
        group_probs = []
        for checkpoint in checkpoints:
            files, labels, probs = predict_probabilities(checkpoint, data_root, split, batch_size)
            if files_ref is None:
                files_ref = files
                labels_ref = labels
            elif [str(path) for path in files] != [str(path) for path in files_ref]:
                raise ValueError(f"File order mismatch for checkpoint {checkpoint}")
            group_probs.append(probs)
        grouped[group] = np.mean(group_probs, axis=0)
    if files_ref is None or labels_ref is None:
        raise ValueError("No ensemble members were provided")
    return files_ref, labels_ref, grouped


def weighted_average(grouped_probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    result = None
    for group, probs in grouped_probs.items():
        weighted = probs * (weights[group] / total)
        result = weighted if result is None else result + weighted
    assert result is not None
    return result


def summarize(labels: np.ndarray, probs: np.ndarray, files: list[Path]) -> dict[str, Any]:
    preds = probs.argmax(axis=1)
    rows = []
    for path, label, pred in zip(files, labels, preds):
        meta = sample_metadata(path)
        rows.append(
            {
                "file": str(path),
                "true": CLASS_NAMES[int(label)],
                "predicted": CLASS_NAMES[int(pred)],
                "correct": bool(label == pred),
                "session_id": str(meta.get("session_id", "")),
                "payload": str(meta.get("payload", "")),
                "snr_db": float(meta.get("snr_db", np.nan)),
                "tx_vga_gain": str(meta.get("tx_vga_gain", "")),
                "rx_gain": str(meta.get("rx_gain", "")),
                "baseband_offset_hz": str(meta.get("baseband_offset_hz", "")),
            }
        )
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(np.mean(preds == labels)),
        "class_recall": class_recall(labels, preds),
        "confusion_matrix": confusion(labels, preds),
        "condition_accuracy": {
            "session_id": condition_accuracy(rows, "session_id"),
            "payload": condition_accuracy(rows, "payload"),
        },
        "top_errors": top_errors(rows),
        "rows": rows,
    }


def class_recall(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        mask = labels == index
        result[name] = float(np.mean(preds[mask] == labels[mask])) if np.any(mask) else float("nan")
    return result


def confusion(labels: np.ndarray, preds: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix.tolist()


def condition_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, ""))].append(bool(row["correct"]))
    return {key: {"accuracy": float(np.mean(values)), "count": len(values)} for key, values in sorted(buckets.items())}


def top_errors(rows: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in rows:
        if row["correct"]:
            continue
        counts[(str(row["true"]), str(row["predicted"]), str(row["session_id"]), str(row["payload"]))] += 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"true": k[0], "predicted": k[1], "session_id": k[2], "payload": k[3], "count": v} for k, v in ranked]


def grid_weights(groups: list[str], step: float) -> list[dict[str, float]]:
    ticks = [round(i * step, 10) for i in range(int(round(1.0 / step)) + 1)]
    candidates = []
    for values in itertools.product(ticks, repeat=len(groups)):
        if abs(sum(values) - 1.0) > 1e-9:
            continue
        if max(values) <= 0:
            continue
        candidates.append(dict(zip(groups, values)))
    return candidates


def optimize_weights(grouped_probs: dict[str, np.ndarray], labels: np.ndarray, step: float) -> dict[str, float]:
    groups = list(grouped_probs)
    best_weights: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    for weights in grid_weights(groups, step):
        probs = weighted_average(grouped_probs, weights)
        preds = probs.argmax(axis=1)
        recalls = class_recall(labels, preds)
        score = (min(recalls.values()), float(np.mean(preds == labels)), float(np.mean(list(recalls.values()))))
        if best_score is None or score > best_score:
            best_score = score
            best_weights = weights
    assert best_weights is not None
    return best_weights


def parse_members(items: list[str]) -> dict[str, list[Path]]:
    members: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Member must use group=path format: {item}")
        group, path = item.split("=", 1)
        members[group].append(Path(path))
    return dict(members)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Experiment 3 Ensemble Evaluation", ""]
    lines.append("## Weights")
    lines.append("")
    lines.append("| group | weight |")
    lines.append("| --- | ---: |")
    for group, weight in result["weights"].items():
        lines.append(f"| {group} | {weight:.3f} |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| split | accuracy | BASK recall | BFSK recall | BPSK recall | worst recall | samples |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split, summary in result["splits"].items():
        recall = summary["class_recall"]
        worst = min(recall.values())
        lines.append(
            f"| {split} | {summary['accuracy']:.4f} | {recall['BASK']:.4f} | {recall['BFSK']:.4f} | {recall['BPSK']:.4f} | {worst:.4f} | {summary['sample_count']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Weights are selected on the validation split and then frozen for test-A/test-B evaluation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_ensemble(
    members: dict[str, list[Path]],
    data_root: str,
    output_dir: str,
    tune_split: str,
    eval_splits: list[str],
    batch_size: int,
    grid_step: float,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    data_root_path = Path(data_root)
    tune_files, tune_labels, tune_grouped = group_probabilities(members, data_root_path, tune_split, batch_size)
    weights = optimize_weights(tune_grouped, tune_labels, grid_step)
    tune_summary = summarize(tune_labels, weighted_average(tune_grouped, weights), tune_files)
    tune_rows = tune_summary.pop("rows")

    result = {
        "members": {group: [str(path) for path in paths] for group, paths in members.items()},
        "tune_split": tune_split,
        "weights": weights,
        "splits": {tune_split: tune_summary},
    }
    write_csv(out / f"{tune_split}_predictions.csv", tune_rows)

    for split in eval_splits:
        files, labels, grouped = group_probabilities(members, data_root_path, split, batch_size)
        summary = summarize(labels, weighted_average(grouped, weights), files)
        rows = summary.pop("rows")
        result["splits"][split] = summary
        write_csv(out / f"{split}_predictions.csv", rows)

    write_json(out / "exp03_ensemble_eval.json", result)
    write_markdown(out / "exp03_ensemble_eval.md", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--output-dir", default="../results/reports/ensemble_eval")
    parser.add_argument("--member", action="append", required=True, help="Ensemble member in group=checkpoint.pt format")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--eval-splits", nargs="+", default=["test_a", "test_b"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()
    evaluate_ensemble(parse_members(args.member), args.data_root, args.output_dir, args.tune_split, args.eval_splits, args.batch_size, args.grid_step)


if __name__ == "__main__":
    main()
