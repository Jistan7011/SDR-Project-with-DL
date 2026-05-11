from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.common import write_json


def run_seed_sweep(
    config_path: str,
    data_root: str,
    output_root: str = "../results",
    seeds: list[int] | None = None,
    model_type: str = "cnn1d",
    eval_split: str = "test",
) -> dict[str, object]:
    seeds = seeds or [42, 43, 44]
    runs = []
    for seed in seeds:
        run_dir = Path(output_root) / f"{model_type}_seed{seed}"
        train_cmd = [
            sys.executable,
            "-m",
            "src.train.train_cnn1d",
            "--config",
            config_path,
            "--preset",
            "train",
            "--seed",
            str(seed),
            "--output-dir",
            str(run_dir),
            "--model-type",
            model_type,
        ]
        eval_cmd = [
            sys.executable,
            "-m",
            "src.train.evaluate",
            "--checkpoint",
            str(run_dir / "checkpoints" / "best.pt"),
            "--config",
            config_path,
            "--data-root",
            data_root,
            "--split",
            eval_split,
            "--output-dir",
            str(run_dir),
        ]
        print("TRAIN:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)
        print("EVAL:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)
        metrics_path = run_dir / "logs" / "eval_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        runs.append(
            {
                "seed": seed,
                "result_dir": str(run_dir),
                "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
                "eval_metrics": metrics,
            }
        )
    summary = {"model_type": model_type, "seeds": seeds, "eval_split": eval_split, "runs": runs}
    write_json(Path(output_root) / f"{model_type}_seed_sweep.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp02.yaml")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--output-root", default="../results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--model-type", default="cnn1d")
    parser.add_argument("--eval-split", default="test")
    args = parser.parse_args()
    run_seed_sweep(args.config, args.data_root, args.output_root, args.seeds, args.model_type, args.eval_split)


if __name__ == "__main__":
    main()
