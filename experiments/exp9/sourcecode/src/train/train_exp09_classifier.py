from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.exp09_fusion_dataset import Exp09FusionDataset
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model
from src.train.train_cnn1d import load_initial_checkpoint, metadata_value
from src.train.train_exp08_multitask_classifier import compute_exp08_metrics, exp08_loss, supervised_contrastive_loss


def train_exp09_classifier(
    config_path: str,
    seed: int | None,
    output_dir: str,
    data_root: str,
    model_type: str,
    preset: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    cfg["model"]["type"] = model_type
    set_seed(int(cfg["project"]["seed"]))
    device = choose_device(str(cfg["train"].get("device", "cpu")))
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])
    is_fusion = model_type.lower().replace("-", "_") in {"fusion_resnet1d_exp9", "exp9_fusion_resnet1d", "fusion_resnet1d_exp9_margin"}
    train_ds, val_ds = build_datasets(data_root, cfg, is_fusion)
    sampler_cfg = cfg["train"].get("sampler", {})
    sampler = build_sampler(train_ds.files, sampler_cfg) if bool(sampler_cfg.get("enabled", False)) else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = build_model(
        model_type,
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    init_checkpoint = str(cfg["train"].get("finetune", {}).get("init_checkpoint", "")).strip()
    if init_checkpoint:
        init_path = resolve_config_relative(config_path, init_checkpoint)
        load_initial_checkpoint(model, init_path, partial=bool(cfg["train"].get("finetune", {}).get("partial_load", True)))
        print(f"initialized_from={init_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))
    loss_cfg = cfg["train"].get("losses", {})
    contrastive_weight = float(loss_cfg.get("contrastive_weight", 0.0))
    if model_type.lower().replace("-", "_").endswith("_margin"):
        contrastive_weight = max(contrastive_weight, float(loss_cfg.get("margin_contrastive_weight", 0.08)))
    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    best_score = -1.0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            xb = move_inputs(xb, device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = exp09_loss(outputs, yb, loss_cfg, contrastive_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        metrics = evaluate_validation(model, val_loader, device)
        score = selection_score(metrics, cfg["train"].get("checkpoint_selection", {}))
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "selection_score": score, **metrics}
        logs.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_worst_recall={metrics['worst_recall']:.4f} "
            f"val_bfsk_recall={metrics['class_recall']['BFSK']:.4f} val_bpsk_recall={metrics['class_recall']['BPSK']:.4f} "
            f"selection={score:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "selection_score": score,
                    "validation_metrics": metrics,
                    "class_names": CLASS_NAMES,
                    "model_type": model_type,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"best_selection_score": best_score, "epochs": logs, "config": cfg})
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    return ckpt_dir / "best.pt"


def build_datasets(data_root: str, cfg: dict[str, Any], is_fusion: bool):
    if is_fusion:
        return Exp09FusionDataset(data_root, "train", preload=bool(cfg["dataset"].get("preload", False))), Exp09FusionDataset(
            data_root, "val", preload=bool(cfg["dataset"].get("preload", False))
        )
    return IQDataset(data_root, "train", feature_mode="precomputed", preload=bool(cfg["dataset"].get("preload", False))), IQDataset(
        data_root, "val", feature_mode="precomputed", preload=bool(cfg["dataset"].get("preload", False))
    )


def exp09_loss(outputs: torch.Tensor | dict[str, torch.Tensor], y: torch.Tensor, cfg: dict[str, Any], contrastive_weight: float) -> torch.Tensor:
    if isinstance(outputs, dict):
        return exp08_loss(outputs, y, cfg, contrastive_weight)
    return nn.functional.cross_entropy(outputs, y)


def evaluate_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(move_inputs(xb, device))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            true.extend(yb.numpy().tolist())
            pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return compute_exp08_metrics(np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64))


def selection_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    return (
        float(cfg.get("accuracy_weight", 0.30)) * float(metrics["accuracy"])
        + float(cfg.get("worst_recall_weight", 0.35)) * float(metrics["worst_recall"])
        + float(cfg.get("bfsk_recall_weight", 0.20)) * float(metrics["class_recall"]["BFSK"])
        + float(cfg.get("bpsk_recall_weight", 0.15)) * float(metrics["class_recall"]["BPSK"])
    )


def build_sampler(files: list[Path], sampler_cfg: dict[str, Any]) -> WeightedRandomSampler:
    keys = [str(key) for key in sampler_cfg.get("balance_keys", ["modulation", "session_id"])]
    max_weight = float(sampler_cfg.get("max_weight", 6.0))
    counts: dict[tuple[str, ...], int] = {}
    buckets: list[tuple[str, ...]] = []
    for path in files:
        meta = sample_metadata(path)
        bucket = tuple(metadata_value(meta, key) for key in keys)
        buckets.append(bucket)
        counts[bucket] = counts.get(bucket, 0) + 1
    weights = np.array([1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    weights = weights / max(float(weights.mean()), 1e-12)
    return WeightedRandomSampler(torch.as_tensor(np.minimum(weights, max_weight), dtype=torch.double), num_samples=len(weights), replacement=True)


def move_inputs(x: torch.Tensor | dict[str, torch.Tensor], device: torch.device):
    if isinstance(x, dict):
        return {key: value.to(device) for key, value in x.items()}
    return x.to(device)


def resolve_config_relative(config_path: str | Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.exists():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp09.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="../results/exp09")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--model-type", default="resnet1d")
    parser.add_argument("--preset", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    train_exp09_classifier(args.config, args.seed, args.output_dir, args.data_root, args.model_type, "smoke" if args.preset == "smoke" else None)


if __name__ == "__main__":
    main()
