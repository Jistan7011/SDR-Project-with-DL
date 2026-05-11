from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def mine_exp08_hard_negatives(
    baseline_checkpoint: str,
    config_path: str,
    data_root: str,
    output_dir: str,
    split: str = "train",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    ckpt = torch.load(baseline_checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=int(model_cfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    feature_mode = str(ckpt_cfg.get("dataset", cfg["dataset"]).get("feature_mode", cfg["dataset"].get("feature_mode", "iq_mag_ifreq_dphase")))
    dataset = IQDataset(data_root, split, feature_mode=feature_mode, preload=bool(cfg["dataset"].get("preload", False)))
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            if isinstance(logits, dict):
                logits = logits["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            labels = yb.numpy()
            for i, (label, guess, prob) in enumerate(zip(labels, pred, probs)):
                path = dataset.files[offset + i]
                meta = sample_metadata(path)
                boundary_error = (label == 0 and guess != 0) or (label != 0 and guess == 0)
                row = {
                    "file": str(path),
                    "file_name": path.name,
                    "expected": CLASS_NAMES[int(label)],
                    "predicted": CLASS_NAMES[int(guess)],
                    "confidence": float(np.max(prob)),
                    "margin": float(np.sort(prob)[-1] - np.sort(prob)[-2]),
                    "boundary_error": bool(boundary_error),
                    "session_id": str(meta.get("session_id", "")),
                    "payload": str(meta.get("payload", "")),
                    "snr_db": float(meta.get("snr_db", np.nan)),
                }
                all_rows.append(row)
                if boundary_error:
                    rows.append(row)
            offset += len(labels)
    result_root = ensure_dir(output_dir)
    write_json(
        result_root / "hard_negative_manifest.json",
        {
            "baseline_checkpoint": baseline_checkpoint,
            "split": split,
            "total_samples": len(all_rows),
            "hard_negative_count": len(rows),
            "hard_negative_rate": len(rows) / max(len(all_rows), 1),
            "hard_negatives": rows,
        },
    )
    write_csv(result_root / "hard_negatives.csv", rows)
    write_csv(result_root / "baseline_train_predictions.csv", all_rows)
    print(f"hard_negative_count={len(rows)} total={len(all_rows)}")
    return {"hard_negative_count": len(rows), "total_samples": len(all_rows)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        values = [csv_escape(row.get(key, "")) for key in keys]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_escape(value: object) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--config", default="../config/config.exp08.yaml")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--output-dir", default="../results/hard_negative_mining")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    mine_exp08_hard_negatives(args.baseline_checkpoint, args.config, args.data_root, args.output_dir, args.split)


if __name__ == "__main__":
    main()
