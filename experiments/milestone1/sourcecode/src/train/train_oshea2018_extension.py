from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.iq_dataset import IQDataset
from src.models.cnn1d import build_model


def train_oshea2018_extension(
    config_path: str,
    data_root: str,
    output_dir: str,
    model_type: str,
    feature_mode: str = "iq_mag_ifreq_dphase",
    input_channels: int = 5,
    loss_type: str = "cross_entropy",
    seed: int | None = None,
    preset: str = "train",
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    cfg.setdefault("model", {})["type"] = model_type
    cfg["model"]["input_channels"] = int(input_channels)
    cfg.setdefault("dataset", {})["root"] = data_root
    cfg["dataset"]["feature_mode"] = feature_mode
    cfg.setdefault("train", {})["loss"] = loss_type
    set_seed(int(cfg["project"]["seed"]))

    device = choose_device(str(cfg["train"].get("device", "cpu")))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    root = Path(data_root)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])
    preload = bool(cfg["dataset"].get("preload", False))
    dropout = float(cfg["model"].get("dropout", 0.3))
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

    train_ds = IQDataset(root, "train", feature_mode=feature_mode, preload=preload)
    val_ds = IQDataset(root, "val", feature_mode=feature_mode, preload=preload)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    model = build_model(model_type, input_channels=input_channels, num_classes=len(CLASS_NAMES), dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 0.001)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    loss_cfg = cfg["train"].get("losses", {})
    contrastive_weight = float(loss_cfg.get("contrastive_weight", 0.0))
    if loss_type == "multitask_margin" or model_type.lower().replace("-", "_").endswith("_margin_5ch"):
        contrastive_weight = max(contrastive_weight, 0.08)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    result_root = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    best_score = -1.0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(xb)
                loss = compute_loss(outputs, yb, loss_type=loss_type, loss_cfg=loss_cfg, contrastive_weight=contrastive_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)

        metrics = evaluate_validation(model, val_loader, device)
        score = selection_score(metrics)
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "selection_score": score, **metrics}
        logs.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_worst_recall={metrics['worst_recall']:.4f} "
            f"val_absorption={metrics['bfsk_bpsk_to_bask_rate']:.4f} selection={score:.4f}"
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
                    "window_size": int(cfg["dataset"].get("window_len", 1024)),
                    "model_type": model_type,
                    "feature_mode": feature_mode,
                    "input_channels": int(input_channels),
                    "loss_type": loss_type,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"best_selection_score": best_score, "epochs": logs, "config": cfg})
    return ckpt_dir / "best.pt"


def compute_loss(outputs: torch.Tensor | dict[str, torch.Tensor], y: torch.Tensor, loss_type: str, loss_cfg: dict[str, Any], contrastive_weight: float) -> torch.Tensor:
    if not isinstance(outputs, dict):
        return F.cross_entropy(outputs, y)
    multiclass = F.cross_entropy(outputs["logits"], y)
    if loss_type == "cross_entropy":
        return multiclass
    nonbask_target = (y != 0).long()
    bask_binary = F.cross_entropy(outputs["bask_binary_logits"], nonbask_target)
    mask = y != 0
    if torch.any(mask):
        bfsk_bpsk_target = (y[mask] == 2).long()
        bfsk_bpsk = F.cross_entropy(outputs["bfsk_bpsk_logits"][mask], bfsk_bpsk_target)
    else:
        bfsk_bpsk = outputs["logits"].sum() * 0.0
    loss = (
        float(loss_cfg.get("multiclass_weight", 1.0)) * multiclass
        + float(loss_cfg.get("bask_binary_weight", 0.45)) * bask_binary
        + float(loss_cfg.get("bfsk_bpsk_weight", 0.55)) * bfsk_bpsk
    )
    if contrastive_weight > 0.0:
        loss = loss + contrastive_weight * supervised_contrastive_loss(outputs["embedding"], y, float(loss_cfg.get("contrastive_temperature", 0.2)))
    return loss


def supervised_contrastive_loss(embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = F.normalize(embedding, dim=1)
    logits = torch.matmul(z, z.T) / max(temperature, 1e-6)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return embedding.sum() * 0.0
    return -(log_prob * positive).sum(dim=1)[valid].div(positive_count[valid]).mean()


def evaluate_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    pin_memory = bool(getattr(loader, "pin_memory", False))
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb.to(device, non_blocking=pin_memory))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            true.extend(yb.numpy().tolist())
            pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return compute_metrics(np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    recalls = {name: float(report[name]["recall"]) for name in CLASS_NAMES}
    nonbask_mask = y_true != 0
    bask_mask = y_true == 0
    bfsk_bpsk_to_bask = float(np.mean(y_pred[nonbask_mask] == 0)) if np.any(nonbask_mask) else 0.0
    bask_to_nonbask = float(np.mean(y_pred[bask_mask] != 0)) if np.any(bask_mask) else 0.0
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "class_recall": recalls,
        "worst_recall": float(min(recalls.values())) if recalls else 0.0,
        "bfsk_bpsk_to_bask_rate": bfsk_bpsk_to_bask,
        "bask_to_nonbask_rate": bask_to_nonbask,
        "confusion_matrix": cm.tolist(),
    }


def selection_score(metrics: dict[str, Any]) -> float:
    return 0.40 * float(metrics["accuracy"]) + 0.45 * float(metrics["worst_recall"]) + 0.15 * (1.0 - float(metrics["bfsk_bpsk_to_bask_rate"]))


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--data-root", default="../data/ota_processed")
    parser.add_argument("--output-dir", default="../results/oshea2018_extension")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--feature-mode", default="iq_mag_ifreq_dphase")
    parser.add_argument("--input-channels", type=int, default=5)
    parser.add_argument("--loss-type", choices=["cross_entropy", "multitask", "multitask_margin"], default="cross_entropy")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preset", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    ckpt = train_oshea2018_extension(
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_type=args.model_type,
        feature_mode=args.feature_mode,
        input_channels=args.input_channels,
        loss_type=args.loss_type,
        seed=args.seed,
        preset=args.preset,
    )
    print(f"Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
