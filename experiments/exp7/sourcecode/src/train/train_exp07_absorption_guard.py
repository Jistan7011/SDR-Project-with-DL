from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.models.absorption_guard import AbsorptionGuardMLP
from src.train.train_cnn1d import train as train_base_classifier


def train_exp07_absorption_guard(
    evidence_root: str = "../data/evidence",
    output_dir: str = "../results/absorption_guard",
    config_path: str = "../config/config.exp07.yaml",
    preset: str | None = None,
    seed: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["project"]["seed"] = int(seed)
    set_seed(int(cfg["project"]["seed"]))
    device = torch.device("cuda" if str(cfg["train"].get("device", "cpu")) == "cuda" and torch.cuda.is_available() else "cpu")
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["guard_epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    train_x, train_y = load_evidence_split(evidence_root, "train")
    val_x, val_y = load_evidence_split(evidence_root, "val")
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    sampler = build_label_sampler(train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = AbsorptionGuardMLP(input_dim=train_x.shape[1], hidden_dim=48, num_classes=len(CLASS_NAMES), dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("guard_learning_rate", 0.001)))
    loss_fn = nn.CrossEntropyLoss()
    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    logs: list[dict[str, Any]] = []
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"guard epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
        metrics = evaluate_guard(model, val_loader, device)
        score = float(0.35 * metrics["accuracy"] + 0.65 * metrics["worst_recall"])
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "selection_score": score, **metrics}
        logs.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_worst_recall={metrics['worst_recall']:.4f} selection={score:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(train_x.shape[1]),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "selection_score": score,
                    "validation_metrics": metrics,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
    write_json(result_root / "logs" / "train_log.json", {"best_selection_score": best_score, "epochs": logs, "config": cfg})
    print(f"Best guard checkpoint: {ckpt_dir / 'best.pt'}")
    return ckpt_dir / "best.pt"


def train_exp07_base_classifier(
    config_path: str = "../config/config.exp07.yaml",
    output_dir: str = "../results/base_classifier",
    preset: str | None = None,
    seed: int | None = None,
    data_root: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    translated = dict(cfg)
    translated["dataset"] = dict(cfg["dataset"])
    translated["dataset"]["root"] = data_root or cfg["dataset"]["root"]
    translated["train"] = dict(cfg["train"])
    translated["train"]["epochs"] = cfg["train"]["base_epochs"]
    temp_config = Path(output_dir) / "config.exp07.base_resolved.yaml"
    ensure_dir(temp_config.parent)
    import yaml

    temp_config.write_text(yaml.safe_dump(translated, sort_keys=False), encoding="utf-8")
    return train_base_classifier(str(temp_config), preset=preset, seed=seed, output_dir=output_dir, model_type="resnet1d")


def load_evidence_split(root: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(root) / f"{split}_evidence.npz", allow_pickle=False)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def build_label_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = labels.astype(np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def evaluate_guard(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device)).cpu()
            true.extend(yb.numpy().tolist())
            pred.extend(logits.argmax(dim=1).numpy().tolist())
    y_true = np.asarray(true, dtype=np.int64)
    y_pred = np.asarray(pred, dtype=np.int64)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    recalls = {name: float(report[name]["recall"]) for name in CLASS_NAMES}
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "class_recall": recalls,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "worst_recall": float(min(recalls.values())) if recalls else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["guard", "base"], default="guard")
    parser.add_argument("--evidence-root", default="../data/evidence")
    parser.add_argument("--output-dir", default="../results/absorption_guard")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    parser.add_argument("--preset", choices=["smoke", "train"], default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    preset = "smoke" if args.preset == "smoke" else None
    if args.mode == "base":
        train_exp07_base_classifier(args.config, args.output_dir, preset, args.seed, args.data_root)
    else:
        train_exp07_absorption_guard(args.evidence_root, args.output_dir, args.config, preset, args.seed)


if __name__ == "__main__":
    main()
