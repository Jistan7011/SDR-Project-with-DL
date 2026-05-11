from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from src.common import ensure_dir, load_config, write_json


STAGE1_CLASSES = ["BASK", "NON_BASK"]
STAGE2_CLASSES = ["BFSK", "BPSK"]


def make_exp06_binary_datasets(config_path: str = "../config/config.exp06.yaml", source_root: str | None = None, output_root: str | None = None) -> Path:
    cfg = load_config(config_path)
    src_root = Path(source_root or cfg["dataset"]["source_root"])
    out_root = ensure_dir(output_root or cfg["dataset"]["stage_root"])
    stage1_root = ensure_dir(out_root / "bask_vs_nonbask")
    stage2_root = ensure_dir(out_root / "bfsk_vs_bpsk")
    for root in [stage1_root, stage2_root]:
        if root.exists():
            shutil.rmtree(root)
        for split in ["train", "val", "test"]:
            ensure_dir(root / split)

    manifest: dict[str, object] = {
        "source_root": str(src_root),
        "stage1": {"class_names": STAGE1_CLASSES, "counts": {"train": 0, "val": 0, "test": 0}, "samples": []},
        "stage2": {"class_names": STAGE2_CLASSES, "counts": {"train": 0, "val": 0, "test": 0}, "samples": []},
    }
    for split in ["train", "val", "test"]:
        for path in sorted((src_root / split).glob("*.npz")):
            data = np.load(path, allow_pickle=False)
            modulation = str(data["modulation"])
            stage1_label = 0 if modulation == "BASK" else 1
            out1 = stage1_root / split / path.name
            save_binary_sample(out1, data, stage1_label, STAGE1_CLASSES[stage1_label], STAGE1_CLASSES)
            manifest["stage1"]["counts"][split] += 1
            manifest["stage1"]["samples"].append({"path": str(out1), "split": split, "source": str(path), "modulation": modulation, "label": stage1_label})

            if modulation in STAGE2_CLASSES:
                stage2_label = STAGE2_CLASSES.index(modulation)
                out2 = stage2_root / split / path.name
                save_binary_sample(out2, data, stage2_label, STAGE2_CLASSES[stage2_label], STAGE2_CLASSES)
                manifest["stage2"]["counts"][split] += 1
                manifest["stage2"]["samples"].append({"path": str(out2), "split": split, "source": str(path), "modulation": modulation, "label": stage2_label})

    validate_no_session_leakage(manifest["stage1"]["samples"])
    validate_no_session_leakage(manifest["stage2"]["samples"])
    write_json(out_root / "manifest_exp06_binary.json", manifest)
    print(f"Exp6 binary datasets written to {out_root}")
    print(f"stage1 counts={manifest['stage1']['counts']}")
    print(f"stage2 counts={manifest['stage2']['counts']}")
    return out_root


def save_binary_sample(out_path: Path, data: np.lib.npyio.NpzFile, label: int, label_name: str, class_names: list[str]) -> None:
    fields = {key: data[key] for key in data.files}
    fields.update(
        {
            "label": np.int64(label),
            "label_name": label_name,
            "binary_class_names": np.asarray(class_names),
            "original_modulation": str(data["modulation"]),
        }
    )
    np.savez(out_path, **fields)


def validate_no_session_leakage(samples: list[dict[str, object]]) -> None:
    seen: dict[str, set[str]] = {}
    for row in samples:
        source = Path(str(row["source"]))
        with np.load(source, allow_pickle=False) as data:
            session = str(data["session_id"])
        seen.setdefault(session, set()).add(str(row["split"]))
    leaked = {session: splits for session, splits in seen.items() if len(splits) > 1}
    if leaked:
        raise SystemExit(f"Session split leakage detected: {leaked}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp06.yaml")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    make_exp06_binary_datasets(args.config, args.source_root, args.output_root)


if __name__ == "__main__":
    main()
