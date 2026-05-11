from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common import ensure_dir, load_config, set_seed, write_json
from src.dataset.binary_iq_dataset import BinaryIQDataset, binary_sample_metadata
from src.models.cnn1d import build_model
from src.train.train_cnn1d import adapt_state_dict_for_model, choose_device


def train_binary_classifier(
    config_path: str,
    stage: str,
    data_root: str | None = None,
    seed: int | None = None,
    output_dir: str = "../results/binary",
    preset: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    set_seed(int(cfg["project"]["seed"]))
    stage_key = normalize_stage(stage)
    class_names = list(cfg["experiment6"][stage_key]["class_names"])
    root = Path(data_root or Path(cfg["dataset"]["stage_root"]) / cfg["experiment6"][stage_key]["dataset_dir"])
    device = choose_device(str(cfg["train"]["device"]))
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    preload = bool(cfg["dataset"].get("preload", False))
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])

    augmentation_cfg = cfg["train"].get("augmentation", {})
    train_ds = BinaryIQDataset(root, "train", augment=bool(augmentation_cfg.get("enabled", False)), augmentation=augmentation_cfg, feature_mode=feature_mode, preload=preload)
    val_ds = BinaryIQDataset(root, "val", feature_mode=feature_mode, preload=preload)
    sampler_cfg = cfg["train"].get("sampler", {})
    sampler = build_binary_sampler(train_ds, sampler_cfg) if bool(sampler_cfg.get("enabled", False)) else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(
        str(cfg["model"].get("type", "resnet1d")),
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=len(class_names),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    init_checkpoint = str(cfg["train"].get("finetune", {}).get("init_checkpoint", "")).strip()
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            init_path = Path(config_path).resolve().parent / init_checkpoint
        ckpt = torch.load(init_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        adapted = adapt_state_dict_for_model(model.state_dict(), state)
        missing, unexpected = model.load_state_dict(adapted, strict=False)
        print(f"initialized_from={init_path}")
        if missing:
            print(f"partial_load_missing_keys={missing}")
        if unexpected:
            print(f"partial_load_unexpected_keys={unexpected}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_cfg = cfg["train"].get("checkpoint_selection", {})
    selection_metric = str(checkpoint_cfg.get("metric", "balanced_score"))
    accuracy_weight = float(checkpoint_cfg.get("accuracy_weight", 0.35))
    best_score = -1.0
    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    log: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"{stage_key} epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
        val_metrics = evaluate_binary_validation(model, val_loader, device, class_names)
        selection_score = score_metrics(val_metrics, selection_metric, accuracy_weight)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "selection_score": selection_score,
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"},
        }
        log.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_worst_recall={val_metrics['worst_recall']:.4f} "
            f"selection={selection_score:.4f}"
        )
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "class_names": class_names,
                    "binary_stage": stage_key,
                    "model_type": str(cfg["model"].get("type", "resnet1d")),
                    "window_size": int(cfg["dataset"]["window_size"]),
                    "config": cfg,
                    "validation_metrics": val_metrics,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"stage": stage_key, "class_names": class_names, "best_selection_score": best_score, "epochs": log, "config": cfg})
    return ckpt_dir / "best.pt"


def normalize_stage(stage: str) -> str:
    key = stage.lower().replace("-", "_")
    if key in {"stage1", "bask_vs_nonbask", "bask_nonbask"}:
        return "stage1"
    if key in {"stage2", "bfsk_vs_bpsk", "bfsk_bpsk"}:
        return "stage2"
    raise ValueError(f"Unsupported stage: {stage}")


def build_binary_sampler(dataset: BinaryIQDataset, sampler_cfg: dict[str, Any]) -> WeightedRandomSampler:
    keys = [str(key) for key in sampler_cfg.get("balance_keys", ["label_name"])]
    max_weight = float(sampler_cfg.get("max_weight", 5.0))
    buckets: list[tuple[str, ...]] = []
    counts: dict[tuple[str, ...], int] = {}
    for path in dataset.files:
        meta = binary_sample_metadata(path)
        bucket = tuple(str(meta.get(key, "")) for key in keys)
        buckets.append(bucket)
        counts[bucket] = counts.get(bucket, 0) + 1
    raw = np.array([1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    raw = raw / max(float(raw.mean()), 1e-12)
    raw = np.minimum(raw, max_weight)
    return WeightedRandomSampler(torch.as_tensor(raw, dtype=torch.double), num_samples=len(raw), replacement=True)


def evaluate_binary_validation(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    recalls = []
    class_recall: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        mask = true == idx
        recall = float(np.mean(pred[mask] == idx)) if np.any(mask) else 0.0
        class_recall[name] = recall
        recalls.append(recall)
    accuracy = float(np.mean(true == pred)) if len(true) else 0.0
    return {
        "accuracy": accuracy,
        "class_recall": class_recall,
        "mean_recall": float(np.mean(recalls)),
        "worst_recall": float(np.min(recalls)),
        "balanced_score": 0.5 * accuracy + 0.5 * float(np.min(recalls)),
        "confusion_matrix": cm.tolist(),
    }


def score_metrics(metrics: dict[str, object], selection_metric: str, accuracy_weight: float) -> float:
    key = selection_metric.lower().replace("-", "_")
    if key == "accuracy":
        return float(metrics["accuracy"])
    if key == "mean_recall":
        return float(metrics["mean_recall"])
    if key == "worst_recall":
        return float(metrics["worst_recall"])
    if key == "balanced_score":
        return accuracy_weight * float(metrics["accuracy"]) + (1.0 - accuracy_weight) * float(metrics["worst_recall"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp06.yaml")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="../results/binary")
    parser.add_argument("--preset", choices=["smoke", "train"], default=None)
    args = parser.parse_args()
    print(f"Best checkpoint: {train_binary_classifier(args.config, args.stage, args.data_root, args.seed, args.output_dir, args.preset)}")


if __name__ == "__main__":
    main()
