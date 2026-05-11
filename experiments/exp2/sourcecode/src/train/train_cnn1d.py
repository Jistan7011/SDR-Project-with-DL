from __future__ import annotations

import argparse
from pathlib import Path

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

    augmentation_cfg = cfg["train"].get("augmentation", {})
    train_ds = IQDataset(root, "train", augment=bool(augmentation_cfg.get("enabled", False)), augmentation=augmentation_cfg, feature_mode=feature_mode)
    val_ds = IQDataset(root, "val", feature_mode=feature_mode)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(
        str(cfg["model"].get("type", "cnn1d")),
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    result_root = Path(output_dir)
    ckpt_dir = ensure_dir(result_root / "checkpoints")
    log: list[dict[str, float]] = []

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

        val_acc = evaluate_accuracy(model, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": total_loss / max(total, 1), "val_accuracy": val_acc}
        log.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val_accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "class_names": CLASS_NAMES,
                    "window_size": int(cfg["dataset"]["window_size"]),
                    "sample_rate": float(cfg["sdr"]["rx_sample_rate"]),
                    "symbol_rate": float(cfg["sdr"]["symbol_rate"]),
                    "model_type": str(cfg["model"].get("type", "cnn1d")),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

    write_json(result_root / "logs" / "train_log.json", {"device": str(device), "best_val_accuracy": best_acc, "epochs": log, "config": cfg})
    return ckpt_dir / "best.pt"


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu()
            correct += int((pred == yb).sum().item())
            total += len(yb)
    return correct / max(total, 1)


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
