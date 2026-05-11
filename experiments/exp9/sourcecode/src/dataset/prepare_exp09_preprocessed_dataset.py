from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.common import ensure_dir, load_config, write_json
from src.signal.canonicalization import canonicalize_iq


PROTOCOL_FIELDS = {"payload", "bits", "crc", "preamble", "sync"}


def prepare_exp09_preprocessed_dataset(source_root: str, output_root: str, config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    src = Path(source_root)
    dst = Path(output_root)
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    dataset_cfg = cfg.get("dataset", {})
    spectral_bins = int(dataset_cfg.get("spectral_bins", 128))
    symbol_samples = int(dataset_cfg.get("symbol_samples", 32))
    split_counts: dict[str, int] = {}
    split_sessions: dict[str, list[str]] = {}
    ensure_dir(dst)
    for split in ["train", "val", "test"]:
        src_split = src / split
        if not src_split.exists():
            raise FileNotFoundError(f"Missing source split: {src_split}")
        dst_split = ensure_clean_dir(dst / split)
        files = sorted(src_split.glob("*.npz"))
        split_counts[split] = len(files)
        sessions: set[str] = set()
        for path in tqdm(files, desc=f"exp09 preprocess:{split}"):
            out_path = dst_split / path.name
            convert_sample(path, out_path, spectral_bins=spectral_bins, symbol_samples=symbol_samples)
            session_id = sample_session_id(out_path)
            if session_id:
                sessions.add(session_id)
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
            "model_input_arrays": ["iq", "spectral_feature", "evidence_feature"],
        },
        "canonicalization": {
            "dc_removal": True,
            "rms_amplitude_normalization": True,
            "coarse_cfo_correction": True,
            "phase_slope_metadata": True,
            "spectral_bins": spectral_bins,
        },
    }
    if leakage:
        raise RuntimeError(f"Session split leakage detected: {leakage}")
    write_json(dst / "manifest_exp09_preprocessed_dataset.json", manifest)
    return manifest


def convert_sample(path: Path, output_path: Path, spectral_bins: int, symbol_samples: int) -> None:
    data = np.load(path, allow_pickle=False)
    iq = data["iq"].astype(np.float32)
    sample_rate = float(data["sample_rate"]) if "sample_rate" in data.files else 160000.0
    result = canonicalize_iq(iq, sample_rate=sample_rate, spectral_bins=spectral_bins, symbol_samples=symbol_samples)
    payload: dict[str, Any] = {}
    for key in data.files:
        if key == "iq":
            continue
        payload[key] = data[key]
    payload.update(
        {
            "iq": result.time_features.astype(np.float32),
            "spectral_feature": result.spectral_feature.astype(np.float32),
            "evidence_feature": result.evidence_feature.astype(np.float32),
            "canonicalization_applied": np.asarray(True),
        }
    )
    for key, value in result.metadata.items():
        payload[f"exp09_{key}"] = np.asarray(value)
    np.savez_compressed(output_path, **payload)


def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


def sample_session_id(path: Path) -> str:
    data = np.load(path, allow_pickle=False)
    if "session_id" in data.files:
        return str(data["session_id"])
    parts = path.name.split("_", 2)
    return "_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "session" else ""


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
    parser.add_argument("--source-root", default="../../exp8/data/processed")
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--config", default="../config/config.exp09.yaml")
    args = parser.parse_args()
    manifest = prepare_exp09_preprocessed_dataset(args.source_root, args.output_root, args.config)
    print(json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
