from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    config_path: str,
    preset: str | None = None,
    seed: int | None = None,
    output_dir: str = "results",
    model_type: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    if model_type is not None:
        cfg["model"]["type"] = model_type
    set_seed(int(cfg["project"]["seed"]))
    device = choose_device(str(cfg["train"]["device"]))
    root = Path(cfg["dataset"]["root"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    preload = bool(cfg["dataset"].get("preload", False))

    augmentation_cfg = cfg["train"].get("augmentation", {})
    train_ds = IQDataset(root, "train", augment=bool(augmentation_cfg.get("enabled", False)), augmentation=augmentation_cfg, feature_mode=feature_mode, preload=preload)
    val_ds = IQDataset(root, "val", feature_mode=feature_mode, preload=preload)
    sampler_cfg = cfg["train"].get("sampler", {})
    train_sampler = build_sampler(train_ds, sampler_cfg) if bool(sampler_cfg.get("enabled", False)) else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(
        str(cfg["model"].get("type", "cnn1d")),
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    finetune_cfg = cfg["train"].get("finetune", {})
    init_checkpoint = str(finetune_cfg.get("init_checkpoint", "")).strip()
    freeze_backbone = bool(finetune_cfg.get("freeze_backbone", False))
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            init_path = Path(config_path).resolve().parent / init_checkpoint
        load_initial_checkpoint(model, init_path)
        print(f"initialized_from={init_path}")
    if freeze_backbone:
        freeze_backbone_parameters(model)
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        print(f"freeze_backbone=true trainable_params={trainable} total_params={total_params}")

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=float(cfg["train"]["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_cfg = cfg["train"].get("checkpoint_selection", {})
    selection_metric = str(checkpoint_cfg.get("metric", "accuracy"))
    accuracy_weight = float(checkpoint_cfg.get("accuracy_weight", 0.5))
    best_score = -1.0
    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    log: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)

        val_metrics = evaluate_validation_metrics(model, val_loader, device)
        selection_score = checkpoint_score(val_metrics, selection_metric, accuracy_weight)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_mean_recall": val_metrics["mean_recall"],
            "val_worst_recall": val_metrics["worst_recall"],
            "val_balanced_score": val_metrics["balanced_score"],
            "val_class_recall": val_metrics["class_recall"],
            "selection_metric": selection_metric,
            "selection_score": selection_score,
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
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "validation_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "window_size": int(cfg["dataset"]["window_size"]),
                    "sample_rate": float(cfg["sdr"]["rx_sample_rate"]),
                    "symbol_rate": float(cfg["sdr"]["symbol_rate"]),
                    "model_type": str(cfg["model"].get("type", "cnn1d")),
                    "init_checkpoint": init_checkpoint or None,
                    "freeze_backbone": freeze_backbone,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

    write_json(
        result_root / "logs" / "train_log.json",
        {
            "device": str(device),
            "selection_metric": selection_metric,
            "best_selection_score": best_score,
            "epochs": log,
            "config": cfg,
        },
    )
    return ckpt_dir / "best.pt"


def build_sampler(dataset: IQDataset, sampler_cfg: dict[str, Any]) -> WeightedRandomSampler:
    keys = [str(key) for key in sampler_cfg.get("balance_keys", ["modulation"])]
    max_weight = float(sampler_cfg.get("max_weight", 5.0))
    counts: dict[tuple[str, ...], int] = {}
    buckets: list[tuple[str, ...]] = []
    for path in dataset.files:
        meta = sample_metadata(path)
        bucket = tuple(metadata_value(meta, key) for key in keys)
        buckets.append(bucket)
        counts[bucket] = counts.get(bucket, 0) + 1

    raw_weights = np.array([1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    raw_weights = raw_weights / max(float(raw_weights.mean()), 1e-12)
    raw_weights = np.minimum(raw_weights, max_weight)
    weights = torch.as_tensor(raw_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def metadata_value(meta: dict[str, Any], key: str) -> str:
    if key == "snr_bin":
        return snr_bin(float(meta.get("snr_db", np.nan)))
    return str(meta.get(key, ""))


def snr_bin(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < -2.0:
        return "< -2 dB"
    if value < 0.0:
        return "-2 to 0 dB"
    if value < 2.0:
        return "0 to 2 dB"
    if value < 4.0:
        return "2 to 4 dB"
    if value < 6.0:
        return "4 to 6 dB"
    return ">= 6 dB"


def load_initial_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)


def freeze_backbone_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def evaluate_validation_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(true == pred)) if len(true) else 0.0
    class_recall: dict[str, float] = {}
    recalls: list[float] = []
    for idx, name in enumerate(CLASS_NAMES):
        mask = true == idx
        recall = float(np.mean(pred[mask] == idx)) if np.any(mask) else 0.0
        class_recall[name] = recall
        recalls.append(recall)
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    worst_recall = float(np.min(recalls)) if recalls else 0.0
    balanced_score = float(0.5 * accuracy + 0.5 * worst_recall)
    return {
        "accuracy": accuracy,
        "class_recall": class_recall,
        "mean_recall": mean_recall,
        "worst_recall": worst_recall,
        "balanced_score": balanced_score,
    }


def checkpoint_score(metrics: dict[str, object], selection_metric: str, accuracy_weight: float) -> float:
    key = selection_metric.lower().replace("-", "_")
    if key in {"accuracy", "val_accuracy"}:
        return float(metrics["accuracy"])
    if key in {"mean_recall", "macro_recall", "val_mean_recall"}:
        return float(metrics["mean_recall"])
    if key in {"worst_recall", "min_recall", "val_worst_recall"}:
        return float(metrics["worst_recall"])
    if key in {"balanced_score", "val_balanced_score"}:
        accuracy = float(metrics["accuracy"])
        worst_recall = float(metrics["worst_recall"])
        return accuracy_weight * accuracy + (1.0 - accuracy_weight) * worst_recall
    raise ValueError(f"Unsupported checkpoint selection metric: {selection_metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--preset", choices=["smoke", "train"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-type", default=None)
    args = parser.parse_args()
    print(f"Best checkpoint: {train(args.config, args.preset, args.seed, args.output_dir, args.model_type)}")


if __name__ == "__main__":
    main()
