from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.absorption_guard import AbsorptionGuardMLP
from src.models.cnn1d import build_model
from src.signal.evidence import (
    analysis_feature_vector,
    classifier_uncertainty,
    compute_signal_evidence,
    decide_unknown_protocol,
    forced_label_from_decision_with_candidates,
)


def evaluate_exp07_unknown_analysis(
    checkpoint: str,
    guard_checkpoint: str | None = None,
    data_root: str | None = None,
    output_dir: str = "../results/unknown_analysis",
    config_path: str = "../config/config.exp07.yaml",
    splits: list[str] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = data_root or cfg["dataset"]["root"]
    result_root = ensure_dir(output_dir)
    base_model = load_base_classifier(checkpoint, cfg)
    guard_model = load_guard(guard_checkpoint) if guard_checkpoint else None
    summary: dict[str, Any] = {"checkpoint": checkpoint, "guard_checkpoint": guard_checkpoint, "splits": {}}
    for split in splits or ["val", "test"]:
        metrics = evaluate_split(base_model, guard_model, cfg, root, split, result_root, max_samples)
        summary["splits"][split] = compact_metrics(metrics)
    write_json(result_root / "exp07_unknown_analysis_summary.json", summary)
    write_summary_md(result_root / "exp07_unknown_analysis_summary.md", summary)
    return summary


def evaluate_split(
    base_model: torch.nn.Module,
    guard_model: torch.nn.Module | None,
    cfg: dict[str, Any],
    data_root: str,
    split: str,
    output_root: Path,
    max_samples: int | None,
) -> dict[str, Any]:
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq_mag_ifreq_dphase"))
    ds = IQDataset(data_root, split, feature_mode=feature_mode, preload=False)
    if max_samples is not None:
        ds.files = ds.files[:max_samples]
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 64)))
    thresholds = dict(cfg["experiment7"].get("evidence_thresholds", {}))
    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_forced: list[int] = []
    file_offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.softmax(base_model(xb), dim=1).cpu().numpy()
            guard_forced: np.ndarray | None = None
            guard_probs: np.ndarray | None = None
            if guard_model is not None:
                guard_inputs = []
                for i in range(len(xb)):
                    channels = ds.load_raw_sample(ds.files[file_offset + i])[0]
                    evidence = compute_signal_evidence(channels)
                    guard_inputs.append(analysis_feature_vector(probs[i], evidence))
                guard_tensor = torch.from_numpy(np.stack(guard_inputs).astype(np.float32))
                guard_probs = torch.softmax(guard_model(guard_tensor), dim=1).cpu().numpy()
                guard_forced = np.argmax(guard_probs, axis=1)
            for i in range(len(xb)):
                path = ds.files[file_offset + i]
                channels = ds.load_raw_sample(path)[0]
                evidence = compute_signal_evidence(channels)
                uncertainty = classifier_uncertainty(probs[i])
                forced_idx = int(guard_forced[i]) if guard_forced is not None else None
                decision = decide_unknown_protocol(probs[i], evidence, thresholds, forced_idx=forced_idx)
                forced_label = forced_label_from_decision_with_candidates(decision.final_decision, decision.candidate_modulations)
                true_idx = int(yb[i])
                forced_idx_final = CLASS_NAMES.index(forced_label)
                y_true.append(true_idx)
                y_forced.append(forced_idx_final)
                meta = sample_metadata(path)
                row = {
                    "file": str(path),
                    "split": split,
                    "expected_modulation": CLASS_NAMES[true_idx],
                    "base_top1": uncertainty["top1"],
                    "base_top2": uncertainty["top2"],
                    "base_top3": uncertainty["top3"],
                    "base_confidence": uncertainty["base_confidence"],
                    "confidence_margin": uncertainty["confidence_margin"],
                    "softmax_entropy": uncertainty["softmax_entropy"],
                    "guard_top1": CLASS_NAMES[int(forced_idx)] if forced_idx is not None else "",
                    "guard_confidence": float(np.max(guard_probs[i])) if guard_probs is not None else "",
                    "forced_3class_prediction": forced_label,
                    "final_decision": decision.final_decision,
                    "candidate_modulations": "|".join(decision.candidate_modulations),
                    "candidate_retains_truth": CLASS_NAMES[true_idx] in decision.candidate_modulations,
                    "is_hard_decision": decision.final_decision in CLASS_NAMES,
                    "hard_decision_correct": decision.final_decision == CLASS_NAMES[true_idx],
                    "bask_absorption_risk": decision.bask_absorption_risk,
                    "unknown_score": decision.unknown_score,
                    "selection_reason": decision.selection_reason,
                    "session_id": meta.get("session_id", ""),
                    "payload": meta.get("payload", ""),
                    "estimated_snr_db": meta.get("estimated_snr_db", meta.get("snr_db", "")),
                    "tx_vga_gain": meta.get("tx_vga_gain", ""),
                    "rx_gain": meta.get("rx_gain", ""),
                    "baseband_offset_hz": meta.get("baseband_offset_hz", ""),
                }
                row.update(evidence)
                rows.append(row)
            file_offset += len(xb)
    metrics = summarize(rows, np.asarray(y_true, dtype=np.int64), np.asarray(y_forced, dtype=np.int64), split)
    write_json(output_root / f"{split}_unknown_metrics.json", {k: v for k, v in metrics.items() if k != "rows"})
    write_csv(output_root / f"{split}_unknown_predictions.csv", rows)
    plot_confusion(output_root / f"{split}_forced_confusion_matrix.png", np.asarray(metrics["forced_confusion_matrix"], dtype=np.int64))
    return metrics


def summarize(rows: list[dict[str, Any]], y_true: np.ndarray, y_forced: np.ndarray, split: str) -> dict[str, Any]:
    report = classification_report(y_true, y_forced, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_forced, labels=list(range(len(CLASS_NAMES))))
    recalls = {name: float(report[name]["recall"]) for name in CLASS_NAMES}
    hard_rows = [row for row in rows if bool(row["is_hard_decision"])]
    bfsk_bpsk_rows = [row for row in rows if row["expected_modulation"] in {"BFSK", "BPSK"}]
    hard_bask_false = [row for row in bfsk_bpsk_rows if row["final_decision"] == "BASK"]
    hard_bask_rows = [row for row in rows if row["final_decision"] == "BASK"]
    return {
        "split": split,
        "sample_count": len(rows),
        "forced_3class_accuracy": float(np.mean(y_true == y_forced)) if len(y_true) else 0.0,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "class_recall": recalls,
        "worst_recall": float(min(recalls.values())) if recalls else 0.0,
        "forced_confusion_matrix": cm.tolist(),
        "hard_decision_rate": rate([bool(row["is_hard_decision"]) for row in rows]),
        "hard_decision_accuracy": rate([bool(row["hard_decision_correct"]) for row in hard_rows]),
        "ambiguous_decision_rate": rate([not bool(row["is_hard_decision"]) for row in rows]),
        "bfsk_bpsk_hard_bask_final_rate": len(hard_bask_false) / max(len(bfsk_bpsk_rows), 1),
        "bfsk_bpsk_candidate_retention_rate": rate([bool(row["candidate_retains_truth"]) for row in bfsk_bpsk_rows]),
        "true_bask_hard_bask_precision": rate([row["expected_modulation"] == "BASK" for row in hard_bask_rows]),
        "final_decision_counts": count_values([str(row["final_decision"]) for row in rows]),
        "condition_accuracy": condition_accuracy(rows),
        "rows": rows,
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_count",
        "forced_3class_accuracy",
        "macro_f1",
        "worst_recall",
        "hard_decision_accuracy",
        "ambiguous_decision_rate",
        "bfsk_bpsk_hard_bask_final_rate",
        "bfsk_bpsk_candidate_retention_rate",
        "true_bask_hard_bask_precision",
        "class_recall",
        "final_decision_counts",
        "forced_confusion_matrix",
    ]
    return {key: metrics[key] for key in keys}


def load_base_classifier(checkpoint: str, cfg: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(resolve_path(checkpoint), map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=len(CLASS_NAMES),
        dropout=float(model_cfg.get("dropout", cfg["model"].get("dropout", 0.35))),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_guard(checkpoint: str | None) -> torch.nn.Module | None:
    if not checkpoint:
        return None
    ckpt = torch.load(resolve_path(checkpoint), map_location="cpu")
    model = AbsorptionGuardMLP(input_dim=int(ckpt["input_dim"]), num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    return candidate.resolve()


def rate(values: list[bool]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def count_values(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def condition_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz"]:
        buckets: dict[str, list[bool]] = {}
        for row in rows:
            buckets.setdefault(str(row.get(key, "")), []).append(row["forced_3class_prediction"] == row["expected_modulation"])
        out[key] = {value: rate(items) for value, items in buckets.items()}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(path: Path, cm: np.ndarray) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xlabel("Forced prediction")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment 7 Unknown-Protocol Analysis",
        "",
        "| split | forced accuracy | macro F1 | worst recall | hard acc | ambiguous rate | BFSK/BPSK hard-BASK | candidate retention | true BASK hard-BASK precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in summary["splits"].items():
        lines.append(
            f"| {split} | {metrics['forced_3class_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['worst_recall']:.4f} | {metrics['hard_decision_accuracy']:.4f} | "
            f"{metrics['ambiguous_decision_rate']:.4f} | {metrics['bfsk_bpsk_hard_bask_final_rate']:.4f} | "
            f"{metrics['bfsk_bpsk_candidate_retention_rate']:.4f} | {metrics['true_bask_hard_bask_precision']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--guard-checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/unknown_analysis")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate_exp07_unknown_analysis(args.checkpoint, args.guard_checkpoint, args.data_root, args.output_dir, args.config, args.splits, args.max_samples)


if __name__ == "__main__":
    main()
