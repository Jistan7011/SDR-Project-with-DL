from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.iq_dataset import IQDataset
from src.models.factory import build_model


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    config_path: str,
    preset: str | None = None,
    model_type: str | None = None,
    freeze_backbone: bool = False,
    init_checkpoint: str | None = None,
    checkpoint_dir: str = "results/checkpoints",
    data_root: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if model_type is not None:
        cfg.setdefault("model", {})["type"] = model_type
    if data_root is not None:
        cfg.setdefault("dataset", {})["root"] = data_root
    set_seed(int(cfg["project"]["seed"]))
    device = choose_device(str(cfg["train"]["device"]))
    root = Path(cfg["dataset"]["root"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["smoke_epochs"] if preset == "smoke" else cfg["train"]["epochs"])

    train_ds = IQDataset(root, "train")
    val_ds = IQDataset(root, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(cfg).to(device)
    if init_checkpoint is not None:
        ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    if freeze_backbone and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(cfg["train"]["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    ckpt_dir = ensure_dir(checkpoint_dir)
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
                    "model_type": str(cfg["model"]["type"]),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

    write_json(
        "results/logs/train_log.json",
        {
            "device": str(device),
            "model_type": str(cfg["model"]["type"]),
            "freeze_backbone": bool(freeze_backbone),
            "init_checkpoint": init_checkpoint,
            "best_val_accuracy": best_acc,
            "epochs": log,
        },
    )
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
    parser.add_argument("--model-type", default=None, help="Override config model.type, e.g. resnet1d or cnn1d")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only classifier layers when supported")
    parser.add_argument("--init-checkpoint", default=None, help="Initialize weights from an existing checkpoint")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--data-root", default=None, help="Override config dataset.root")
    args = parser.parse_args()
    print(
        "Best checkpoint: "
        f"{train(args.config, args.preset, args.model_type, args.freeze_backbone, args.init_checkpoint, args.checkpoint_dir, args.data_root)}"
    )


if __name__ == "__main__":
    main()
