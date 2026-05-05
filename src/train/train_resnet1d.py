from __future__ import annotations

import argparse

from src.train.train_cnn1d import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--preset", choices=["smoke", "train"], default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    checkpoint = train(
        config_path=args.config,
        preset=args.preset,
        model_type="resnet1d",
        freeze_backbone=args.freeze_backbone,
        init_checkpoint=args.init_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
