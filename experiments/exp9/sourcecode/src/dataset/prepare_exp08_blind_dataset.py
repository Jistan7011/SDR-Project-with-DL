from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from src.common import ensure_dir, write_json


def prepare_exp08_blind_dataset(source_root: str, output_root: str, copy_files: bool = True) -> dict[str, object]:
    src = Path(source_root)
    dst = Path(output_root)
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    split_counts: dict[str, int] = {}
    split_sessions: dict[str, list[str]] = {}
    ensure_dir(dst)
    for split in ["train", "val", "test"]:
        src_split = src / split
        if not src_split.exists():
            raise FileNotFoundError(f"Missing source split: {src_split}")
        dst_split = ensure_dir(dst / split)
        files = sorted(src_split.glob("*.npz"))
        split_counts[split] = len(files)
        sessions = set()
        for path in tqdm(files, desc=f"exp08 prepare:{split}"):
            session_id = path.name.split("_", 2)[0] + "_" + path.name.split("_", 2)[1] if path.name.startswith("session_") else ""
            if session_id:
                sessions.add(session_id)
            target = dst_split / path.name
            if copy_files and (not target.exists() or target.stat().st_size != path.stat().st_size):
                shutil.copy2(path, target)
        split_sessions[split] = sorted(sessions)
    leakage = find_split_leakage(split_sessions)
    manifest = {
        "source_root": str(src),
        "output_root": str(dst),
        "split_counts": split_counts,
        "split_sessions": split_sessions,
        "split_leakage": leakage,
        "protocol_blind_inputs": {
            "uses_payload_as_input": False,
            "uses_crc_as_input": False,
            "uses_preamble_sync_as_input": False,
        },
    }
    if leakage:
        raise RuntimeError(f"Session split leakage detected: {leakage}")
    write_json(dst / "manifest_exp08_blind_dataset.json", manifest)
    return manifest


def find_split_leakage(split_sessions: dict[str, list[str]]) -> dict[str, list[str]]:
    leakage: dict[str, list[str]] = {}
    names = list(split_sessions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(split_sessions[left]) & set(split_sessions[right]))
            if overlap:
                leakage[f"{left}_vs_{right}"] = overlap
    return leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="../../exp7/data/processed_retrained")
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    manifest = prepare_exp08_blind_dataset(args.source_root, args.output_root, copy_files=not args.no_copy)
    print(json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
