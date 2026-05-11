from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_name(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "session":
        raise ValueError(f"Unexpected sample filename: {path.name}")
    return f"{parts[0]}_{parts[1]}", parts[2].upper()


def collect_by_session_mod(split_dir: Path) -> dict[str, dict[str, list[Path]]]:
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(split_dir.glob("*.npz")):
        session_id, modulation = parse_name(path)
        grouped[session_id][modulation].append(path)
    return grouped


def make_split(
    source_root: Path,
    output_root: Path,
    split: str,
    per_session_class_limit: int,
    min_per_session_class: int,
) -> dict[str, object]:
    grouped = collect_by_session_mod(source_root / split)
    out_split = output_root / split
    out_split.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    session_counts: dict[str, dict[str, int]] = {}
    excluded: dict[str, dict[str, int]] = {}
    used_sessions: list[str] = []

    for session_id in sorted(grouped):
        available = {mod: len(grouped[session_id].get(mod, [])) for mod in ("BASK", "BFSK", "BPSK")}
        min_available = min(available.values())
        if min_available < min_per_session_class:
            excluded[session_id] = available
            continue
        take = min(per_session_class_limit, min_available)
        used_sessions.append(session_id)
        session_counts[session_id] = {}
        for mod in ("BASK", "BFSK", "BPSK"):
            selected = grouped[session_id][mod][:take]
            session_counts[session_id][mod] = len(selected)
            counts[mod] += len(selected)
            for idx, src in enumerate(selected):
                dst = out_split / f"{session_id}_{mod.lower()}_{idx:06d}.npz"
                link_or_copy(src, dst)

    return {
        "split": split,
        "per_session_class_limit": per_session_class_limit,
        "min_per_session_class": min_per_session_class,
        "used_sessions": used_sessions,
        "excluded_sessions": excluded,
        "counts": dict(counts),
        "session_counts": session_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-per-session-class", type=int, default=9000)
    parser.add_argument("--val-per-session-class", type=int, default=12000)
    parser.add_argument("--test-per-session-class", type=int, default=10000)
    parser.add_argument("--train-min-per-session-class", type=int, default=5000)
    parser.add_argument("--val-min-per-session-class", type=int, default=5000)
    parser.add_argument("--test-min-per-session-class", type=int, default=5000)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    ensure_empty_dir(output_root)

    reports = [
        make_split(source_root, output_root, "train", args.train_per_session_class, args.train_min_per_session_class),
        make_split(source_root, output_root, "val", args.val_per_session_class, args.val_min_per_session_class),
        make_split(source_root, output_root, "test", args.test_per_session_class, args.test_min_per_session_class),
    ]
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "method": "session_class_balanced_hardlink_subset",
        "reports": reports,
    }
    (output_root / "balanced_subset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
