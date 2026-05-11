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


def prepare_oshea2018_rf_preprocessed(source_root: str, output_root: str, config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    src = Path(source_root)
    dst = Path(output_root)
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    dataset_cfg = cfg.get("dataset", {})
    spectral_bins = int(dataset_cfg.get("spectral_bins", 128))
    symbol_samples = int(dataset_cfg.get("symbol_samples", 32))
    sample_rate = float(cfg.get("sdr", {}).get("rx_sample_rate", cfg.get("sdr", {}).get("sample_rate", 2_400_000)))

    ensure_dir(dst)
    split_counts: dict[str, int] = {}
    split_sessions: dict[str, list[str]] = {}
    for split in ["train", "val", "test"]:
        src_split = src / split
        if not src_split.exists():
            raise FileNotFoundError(f"Missing source split: {src_split}")
        dst_split = ensure_clean_dir(dst / split)
        files = sorted(src_split.glob("*.npz"))
        split_counts[split] = len(files)
        sessions: set[str] = set()
        for path in tqdm(files, desc=f"oshea2018 rf preprocess:{split}"):
            out_path = dst_split / path.name
            session = convert_sample(path, out_path, sample_rate=sample_rate, spectral_bins=spectral_bins, symbol_samples=symbol_samples)
            if session:
                sessions.add(session)
        split_sessions[split] = sorted(sessions)

    leakage = find_split_leakage(split_sessions)
    manifest = {
        "source_root": str(src),
        "output_root": str(dst),
        "split_counts": split_counts,
        "split_sessions": split_sessions,
        "split_leakage": leakage,
        "model_input_arrays": ["iq"],
        "iq_shape": [5, int(dataset_cfg.get("window_len", 1024))],
        "rf_preprocessing": {
            "dc_removal": True,
            "rms_amplitude_normalization": True,
            "coarse_phase_slope_correction": True,
            "feature_channels": ["I", "Q", "magnitude", "instantaneous_frequency", "differential_phase"],
            "spectral_bins": spectral_bins,
            "symbol_samples": symbol_samples,
        },
    }
    if leakage:
        raise RuntimeError(f"Session split leakage detected: {leakage}")
    write_json(dst / "manifest_oshea2018_rf_preprocessed.json", manifest)
    return manifest


def convert_sample(path: Path, output_path: Path, sample_rate: float, spectral_bins: int, symbol_samples: int) -> str:
    data = np.load(path, allow_pickle=False)
    result = canonicalize_iq(data["iq"].astype(np.float32), sample_rate=sample_rate, spectral_bins=spectral_bins, symbol_samples=symbol_samples)
    payload: dict[str, Any] = {}
    for key in data.files:
        if key == "iq":
            continue
        payload[key] = data[key]
    payload["iq"] = result.time_features.astype(np.float32)
    payload["spectral_feature"] = result.spectral_feature.astype(np.float32)
    payload["rf_preprocessing_applied"] = np.asarray(True)
    for key, value in result.metadata.items():
        payload[f"rf_{key}"] = np.asarray(value)
    np.savez_compressed(output_path, **payload)
    if "session_id" in data.files:
        return str(data["session_id"])
    parts = path.name.split("_", 2)
    return "_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "session" else ""


def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


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
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--source-root", default="../data/ota_processed")
    parser.add_argument("--output-root", default="../data/ota_rf_preprocessed")
    args = parser.parse_args()
    manifest = prepare_oshea2018_rf_preprocessed(args.source_root, args.output_root, args.config)
    print(json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
