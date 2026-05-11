from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model
from src.train.train_cnn1d import load_initial_checkpoint, metadata_value


def train_exp08_multitask_classifier(
    config_path: str,
    seed: int | None = None,
    output_dir: str = "../results/exp08_multitask",
    data_root: str | None = None,
    model_type: str | None = None,
    preset: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    if model_type is not None:
        cfg["model"]["type"] = model_type
    set_seed(int(cfg["project"]["seed"]))
    device = choose_device(str(cfg["train"].get("device", "cpu")))
    root = Path(data_root or cfg["dataset"]["root"])
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq_mag_ifreq_dphase"))
    preload = bool(cfg["dataset"].get("preload", False))
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])

    augmentation_cfg = cfg["train"].get("augmentation", {})
    train_ds = IQDataset(root, "train", augment=bool(augmentation_cfg.get("enabled", False)), augmentation=augmentation_cfg, feature_mode=feature_mode, preload=preload)
    val_ds = IQDataset(root, "val", feature_mode=feature_mode, preload=preload)
    sampler_cfg = cfg["train"].get("sampler", {})
    sampler = build_exp08_sampler(train_ds, sampler_cfg, config_path) if bool(sampler_cfg.get("enabled", False)) else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(
        str(cfg["model"].get("type", "multitask_resnet1d")),
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    finetune_cfg = cfg["train"].get("finetune", {})
    init_checkpoint = str(finetune_cfg.get("init_checkpoint", "")).strip()
    if init_checkpoint:
        init_path = resolve_config_relative(config_path, init_checkpoint)
        load_initial_checkpoint(model, init_path, partial=bool(finetune_cfg.get("partial_load", True)))
        print(f"initialized_from={init_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"].get("weight_decay", 0.01)))
    loss_cfg = cfg["train"].get("losses", {})
    contrastive_weight = float(loss_cfg.get("contrastive_weight", 0.0))
    if str(cfg["model"].get("type", "")).lower().replace("-", "_").endswith("_margin"):
        contrastive_weight = max(contrastive_weight, 0.08)

    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    best_score = -1.0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = exp08_loss(outputs, yb, loss_cfg, contrastive_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)

        metrics = evaluate_exp08_validation(model, val_loader, device)
        score = selection_score(metrics, cfg["train"].get("checkpoint_selection", {}))
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "selection_score": score, **metrics}
        logs.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_worst_recall={metrics['worst_recall']:.4f} "
            f"val_nonbask_boundary_score={metrics['nonbask_boundary_score']:.4f} selection={score:.4f}"
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
                    "window_size": int(cfg["dataset"]["window_size"]),
                    "sample_rate": float(cfg["sdr"]["rx_sample_rate"]),
                    "symbol_rate": float(cfg["sdr"]["symbol_rate"]),
                    "model_type": str(cfg["model"].get("type", "multitask_resnet1d")),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"best_selection_score": best_score, "epochs": logs, "config": cfg})
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    return ckpt_dir / "best.pt"


def exp08_loss(outputs: dict[str, torch.Tensor], y: torch.Tensor, cfg: dict[str, Any], contrastive_weight: float) -> torch.Tensor:
    multiclass = F.cross_entropy(outputs["logits"], y)
    nonbask_target = (y != 0).long()
    bask_binary = F.cross_entropy(outputs["bask_binary_logits"], nonbask_target)
    mask = y != 0
    if torch.any(mask):
        bfsk_bpsk_target = (y[mask] == 2).long()
        bfsk_bpsk = F.cross_entropy(outputs["bfsk_bpsk_logits"][mask], bfsk_bpsk_target)
    else:
        bfsk_bpsk = outputs["logits"].sum() * 0.0
    loss = (
        float(cfg.get("multiclass_weight", 1.0)) * multiclass
        + float(cfg.get("bask_binary_weight", 0.45)) * bask_binary
        + float(cfg.get("bfsk_bpsk_weight", 0.55)) * bfsk_bpsk
    )
    if contrastive_weight > 0:
        loss = loss + contrastive_weight * supervised_contrastive_loss(outputs["embedding"], y, float(cfg.get("contrastive_temperature", 0.2)))
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


def evaluate_exp08_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb.to(device))
            true.extend(yb.numpy().tolist())
            pred.extend(outputs["logits"].argmax(dim=1).cpu().numpy().tolist())
    return compute_exp08_metrics(np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64))


def compute_exp08_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    recalls = {name: float(report[name]["recall"]) for name in CLASS_NAMES}
    nonbask_mask = y_true != 0
    bfsk_bpsk_to_bask = float(np.mean(y_pred[nonbask_mask] == 0)) if np.any(nonbask_mask) else 0.0
    bask_mask = y_true == 0
    bask_to_nonbask = float(np.mean(y_pred[bask_mask] != 0)) if np.any(bask_mask) else 0.0
    nonbask_boundary_score = float(1.0 - 0.5 * (bfsk_bpsk_to_bask + bask_to_nonbask))
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "class_recall": recalls,
        "worst_recall": float(min(recalls.values())) if recalls else 0.0,
        "bfsk_bpsk_to_bask_rate": bfsk_bpsk_to_bask,
        "bask_to_nonbask_rate": bask_to_nonbask,
        "nonbask_boundary_score": nonbask_boundary_score,
        "confusion_matrix": cm.tolist(),
    }


def selection_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    key = str(cfg.get("metric", "exp08_balanced")).lower().replace("-", "_")
    if key == "accuracy":
        return float(metrics["accuracy"])
    if key == "worst_recall":
        return float(metrics["worst_recall"])
    return (
        float(cfg.get("accuracy_weight", 0.35)) * float(metrics["accuracy"])
        + float(cfg.get("worst_recall_weight", 0.45)) * float(metrics["worst_recall"])
        + float(cfg.get("nonbask_boundary_weight", 0.20)) * float(metrics["nonbask_boundary_score"])
    )


def build_exp08_sampler(dataset: IQDataset, sampler_cfg: dict[str, Any], config_path: str) -> WeightedRandomSampler:
    keys = [str(key) for key in sampler_cfg.get("balance_keys", ["modulation"])]
    max_weight = float(sampler_cfg.get("max_weight", 6.0))
    hard_negative_weight = float(sampler_cfg.get("hard_negative_weight", 2.0))
    hard_negative_files = load_hard_negative_files(sampler_cfg, config_path)
    counts: dict[tuple[str, ...], int] = {}
    buckets: list[tuple[str, ...]] = []
    for path in dataset.files:
        meta = sample_metadata(path)
        bucket = tuple(metadata_value(meta, key) for key in keys)
        buckets.append(bucket)
        counts[bucket] = counts.get(bucket, 0) + 1
    raw_weights = np.array([1.0 / counts[bucket] for bucket in buckets], dtype=np.float64)
    raw_weights = raw_weights / max(float(raw_weights.mean()), 1e-12)
    if hard_negative_files:
        for idx, path in enumerate(dataset.files):
            if path.name in hard_negative_files:
                raw_weights[idx] *= hard_negative_weight
    raw_weights = np.minimum(raw_weights, max_weight)
    return WeightedRandomSampler(torch.as_tensor(raw_weights, dtype=torch.double), num_samples=len(raw_weights), replacement=True)


def load_hard_negative_files(sampler_cfg: dict[str, Any], config_path: str) -> set[str]:
    manifest = str(sampler_cfg.get("hard_negative_manifest", "")).strip()
    if not manifest:
        return set()
    path = resolve_config_relative(config_path, manifest)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["file_name"]) for item in data.get("hard_negatives", [])}


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
    parser.add_argument("--config", default="../config/config.exp08.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="../results/exp08_multitask")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--preset", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    preset = "smoke" if args.preset == "smoke" else None
    train_exp08_multitask_classifier(args.config, args.seed, args.output_dir, args.data_root, args.model_type, preset)


if __name__ == "__main__":
    main()
