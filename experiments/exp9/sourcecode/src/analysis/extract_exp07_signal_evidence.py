from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model
from src.signal.evidence import EVIDENCE_NAMES, analysis_feature_vector, classifier_uncertainty, compute_signal_evidence


def extract_exp07_signal_evidence(
    checkpoint: str,
    data_root: str | None = None,
    output_root: str = "../data/evidence",
    config_path: str = "../config/config.exp07.yaml",
    splits: list[str] | None = None,
    max_samples: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    root = data_root or cfg["dataset"]["root"]
    out = ensure_dir(output_root)
    model = load_classifier(checkpoint, cfg)
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq_mag_ifreq_dphase"))
    batch_size = int(cfg["train"].get("batch_size", 64))
    split_names = splits or ["train", "val", "test"]
    manifest: dict[str, Any] = {"checkpoint": checkpoint, "data_root": str(root), "splits": {}}
    for split in split_names:
        ds = IQDataset(root, split, feature_mode=feature_mode, preload=False)
        if max_samples is not None:
            ds.files = ds.files[:max_samples]
        loader = DataLoader(ds, batch_size=batch_size)
        rows: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        labels: list[int] = []
        probs_rows: list[np.ndarray] = []
        file_offset = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb).cpu()
                probs = torch.softmax(logits, dim=1).numpy()
                for i in range(len(xb)):
                    path = ds.files[file_offset + i]
                    channels = ds.load_raw_sample(path)[0]
                    evidence = compute_signal_evidence(channels)
                    uncertainty = classifier_uncertainty(probs[i])
                    vec = analysis_feature_vector(probs[i], evidence)
                    meta = sample_metadata(path)
                    row = {
                        "file": str(path),
                        "split": split,
                        "expected_modulation": CLASS_NAMES[int(yb[i])],
                        "base_top1": uncertainty["top1"],
                        "base_top2": uncertainty["top2"],
                        "base_top3": uncertainty["top3"],
                        "base_confidence": uncertainty["base_confidence"],
                        "confidence_margin": uncertainty["confidence_margin"],
                        "softmax_entropy": uncertainty["softmax_entropy"],
                        "session_id": meta.get("session_id", ""),
                        "payload": meta.get("payload", ""),
                        "estimated_snr_db": meta.get("estimated_snr_db", meta.get("snr_db", "")),
                        "tx_vga_gain": meta.get("tx_vga_gain", ""),
                        "rx_gain": meta.get("rx_gain", ""),
                        "baseband_offset_hz": meta.get("baseband_offset_hz", ""),
                    }
                    row.update(evidence)
                    rows.append(row)
                    features.append(vec)
                    labels.append(int(yb[i]))
                    probs_rows.append(probs[i].astype(np.float32))
                file_offset += len(xb)
        x = np.stack(features).astype(np.float32) if features else np.zeros((0, 11), dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        p = np.stack(probs_rows).astype(np.float32) if probs_rows else np.zeros((0, 3), dtype=np.float32)
        np.savez(out / f"{split}_evidence.npz", features=x, labels=y, probs=p, evidence_names=np.asarray(EVIDENCE_NAMES), class_names=np.asarray(CLASS_NAMES))
        write_csv(out / f"{split}_evidence.csv", rows)
        manifest["splits"][split] = {"count": int(len(rows)), "npz": str(out / f"{split}_evidence.npz"), "csv": str(out / f"{split}_evidence.csv")}
    write_json(out / "manifest_exp07_evidence.json", manifest)
    print(f"Exp7 evidence written to {out}")
    return out


def load_classifier(checkpoint: str, cfg: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(resolve_checkpoint(checkpoint, cfg), map_location="cpu")
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


def resolve_checkpoint(checkpoint: str, cfg: dict[str, Any]) -> Path:
    path = Path(checkpoint)
    if path.exists():
        return path
    config_path = Path(str(cfg.get("_config_path", ".")))
    return (config_path.parent / checkpoint).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default="../data/evidence")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    extract_exp07_signal_evidence(args.checkpoint, args.data_root, args.output_root, args.config, args.splits, args.max_samples)


if __name__ == "__main__":
    main()
