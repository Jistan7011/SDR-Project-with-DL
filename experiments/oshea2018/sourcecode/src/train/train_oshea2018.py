from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.iq_dataset import IQDataset
from src.models.cnn1d import build_model


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def freeze_except_classifier(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def load_checkpoint(model: nn.Module, checkpoint: str | Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)


def make_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    lr = float(cfg["train"]["learning_rate"])
    optimizer_name = str(cfg["train"].get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    pin_memory = bool(getattr(loader, "pin_memory", False))
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device, non_blocking=pin_memory))
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(true == pred)) if len(true) else 0.0
    recalls: list[float] = []
    class_recall: dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = true == idx
        recall = float(np.mean(pred[mask] == idx)) if np.any(mask) else 0.0
        class_recall[name] = recall
        recalls.append(recall)
    return {
        "accuracy": accuracy,
        "class_recall": class_recall,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "worst_recall": float(np.min(recalls)) if recalls else 0.0,
    }


def train_oshea2018(
    config_path: str,
    data_root: str | None = None,
    output_dir: str = "../results/oshea2018",
    model_type: str | None = None,
    seed: int | None = None,
    preset: str = "train",
    init_checkpoint: str | None = None,
    freeze_backbone: bool = False,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    if model_type is not None:
        cfg["model"]["type"] = model_type
    if data_root is not None:
        cfg["dataset"]["root"] = data_root
    set_seed(int(cfg["project"]["seed"]))
    device = choose_device(str(cfg["train"]["device"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    root = Path(cfg["dataset"]["root"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])
    preload = bool(cfg["dataset"].get("preload", False))
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", device.type == "cuda"))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    use_amp = bool(cfg["train"].get("amp", device.type == "cuda"))

    train_ds = IQDataset(root, "train", feature_mode="iq", preload=preload)
    val_ds = IQDataset(root, "val", feature_mode="iq", preload=preload)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    model = build_model(
        str(cfg["model"]["type"]),
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    if init_checkpoint:
        load_checkpoint(model, init_checkpoint)
        print(f"initialized_from={init_checkpoint}")
    if freeze_backbone:
        freeze_except_classifier(model)
        print("freeze_backbone=true trainable=classifier")
    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    result_root = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    best_acc = -1.0
    log: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
        val_metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_worst_recall": val_metrics["worst_recall"],
            "val_class_recall": val_metrics["class_recall"],
        }
        log.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={row['val_accuracy']:.4f} val_worst_recall={row['val_worst_recall']:.4f}"
        )
        if float(val_metrics["accuracy"]) > best_acc:
            best_acc = float(val_metrics["accuracy"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "class_names": CLASS_NAMES,
                    "model_type": str(cfg["model"]["type"]),
                    "window_size": int(cfg["dataset"]["window_len"]),
                    "config": cfg,
                    "init_checkpoint": init_checkpoint,
                    "freeze_backbone": freeze_backbone,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"best_val_accuracy": best_acc, "epochs": log, "config": cfg})
    return ckpt_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/oshea2018")
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preset", choices=["smoke", "train"], default="train")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()
    ckpt = train_oshea2018(
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_type=args.model_type,
        seed=args.seed,
        preset=args.preset,
        init_checkpoint=args.init_checkpoint,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
