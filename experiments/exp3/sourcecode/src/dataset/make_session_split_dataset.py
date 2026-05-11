from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from src.dataset.iq_dataset import sample_metadata


DEFAULT_SOURCE_SPLITS = ["train", "val", "test_a", "test_b", "test"]


def parse_sessions(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def collect_files(source_root: Path, source_splits: list[str]) -> list[Path]:
    files: list[Path] = []
    for split in source_splits:
        split_dir = source_root / split
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*.npz")))
    return files


def make_session_split_dataset(
    source_root: str,
    output_root: str,
    train_sessions: set[str],
    val_sessions: set[str],
    test_sessions: set[str],
    source_splits: list[str],
    copy_mode: str,
) -> dict[str, int]:
    source = Path(source_root)
    output = Path(output_root)
    if output.exists():
        shutil.rmtree(output)
    for split in ["train", "val", "test"]:
        (output / split).mkdir(parents=True, exist_ok=True)

    assignment = {}
    for session_id in train_sessions:
        assignment[session_id] = "train"
    for session_id in val_sessions:
        if session_id in assignment:
            raise ValueError(f"Session appears in multiple splits: {session_id}")
        assignment[session_id] = "val"
    for session_id in test_sessions:
        if session_id in assignment:
            raise ValueError(f"Session appears in multiple splits: {session_id}")
        assignment[session_id] = "test"

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0, "duplicates": 0}
    for path in tqdm(collect_files(source, source_splits), desc="split-by-session"):
        meta = sample_metadata(path)
        session_id = str(meta.get("session_id", ""))
        split = assignment.get(session_id)
        if split is None:
            counts["skipped"] += 1
            continue
        target = output / split / path.name
        if target.exists():
            counts["duplicates"] += 1
            continue
        if copy_mode == "hardlink":
            try:
                target.hardlink_to(path.resolve())
            except OSError:
                shutil.copy2(path, target)
        elif copy_mode == "copy":
            shutil.copy2(path, target)
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")
        counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="../data/processed")
    parser.add_argument("--output-root", default="../data/processed_domain_adapt")
    parser.add_argument("--train-sessions", required=True, help="Comma-separated session IDs")
    parser.add_argument("--val-sessions", required=True, help="Comma-separated session IDs")
    parser.add_argument("--test-sessions", required=True, help="Comma-separated session IDs")
    parser.add_argument("--source-splits", nargs="+", default=DEFAULT_SOURCE_SPLITS)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()
    counts = make_session_split_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        train_sessions=parse_sessions(args.train_sessions),
        val_sessions=parse_sessions(args.val_sessions),
        test_sessions=parse_sessions(args.test_sessions),
        source_splits=args.source_splits,
        copy_mode=args.copy_mode,
    )
    print(counts)


if __name__ == "__main__":
    main()
