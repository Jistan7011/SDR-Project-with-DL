from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from src.common import ensure_dir, load_config, write_json


def import_exp07_unknown_protocol_windows(
    source_root: str | None = None,
    output_root: str = "../data/processed",
    config_path: str = "../config/config.exp07.yaml",
) -> Path:
    cfg = load_config(config_path)
    src = Path(source_root or cfg["dataset"]["reference_root"])
    if not src.exists():
        raise SystemExit(f"Missing source dataset: {src}")
    out = Path(output_root)
    if out.exists():
        shutil.rmtree(out)
    counters: dict[str, int] = {}
    manifest: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_src = src / split
        split_out = ensure_dir(out / split)
        files = sorted(split_src.glob("*.npz"))
        counters[split] = len(files)
        for path in tqdm(files, desc=f"exp07-import:{split}"):
            target = split_out / path.name
            shutil.copy2(path, target)
            manifest.append({"split": split, "source": str(path), "path": str(target)})
    write_json(
        out / "manifest_exp07_unknown_protocol.json",
        {
            "source_root": str(src),
            "sample_count": sum(counters.values()),
            "counts": counters,
            "unknown_protocol_note": "Payload/frame/CRC metadata is retained only for ground-truth evaluation and is not used as model input.",
            "samples": manifest,
        },
    )
    print(f"Imported exp7 unknown-protocol windows to {out}: {counters}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    args = parser.parse_args()
    import_exp07_unknown_protocol_windows(args.source_root, args.output_root, args.config)


if __name__ == "__main__":
    main()
