from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.split_by_session import split_name_for_class
from src.signal.awgn import add_awgn
from src.signal.frame import make_frame
from src.signal.modulate import modulate_bits
from src.signal.processing import iq_to_channels


def generate_dataset(config_path: str, preset: str, output_root: str | None = None) -> Path:
    cfg = load_config(config_path)
    set_seed(int(cfg["project"]["seed"]))
    root = ensure_dir(output_root or cfg["dataset"]["root"])
    for split in ("train", "val", "test"):
        ensure_dir(root / split)

    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    sps = max(1, int(round(sample_rate / symbol_rate)))
    window_size = int(cfg["dataset"]["window_size"])
    payloads = list(cfg["modulation"]["payloads"])
    snrs = list(cfg["awgn"]["snr_list"])
    samples_per_class = int(cfg["dataset"]["samples_per_class"][preset])
    total_sessions = samples_per_class * len(CLASS_NAMES)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    session = 0
    for modulation in CLASS_NAMES:
        for class_sample in tqdm(range(samples_per_class), desc=f"{preset}:{modulation}"):
            payload = str(rng.choice(payloads))
            snr_db = float(rng.choice(snrs))
            bits = make_frame(payload)
            clean = modulate_bits(modulation, bits, sps, sample_rate)
            noisy = add_awgn(clean, snr_db, rng)
            channels = iq_to_channels(noisy, window_size)
            split = split_name_for_class(class_sample, samples_per_class, float(cfg["dataset"]["train_ratio"]), float(cfg["dataset"]["val_ratio"]))
            name = f"sample_{counters[split]:06d}.npz"
            path = root / split / name
            np.savez(
                path,
                iq=channels,
                raw_iq=noisy.astype(np.complex64),
                modulation=modulation,
                payload=payload,
                bits=bits.astype(np.uint8),
                snr_db=snr_db,
                sample_rate=sample_rate,
                symbol_rate=symbol_rate,
                center_freq=int(cfg["sdr"]["center_freq"]),
                session_id=f"session_{session:06d}",
            )
            manifest.append({"path": str(path), "split": split, "modulation": modulation, "payload": payload, "snr_db": snr_db})
            counters[split] += 1
            session += 1

    write_json(root / f"manifest_{preset}.json", {"preset": preset, "counts": counters, "samples": manifest})
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--preset", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    root = generate_dataset(args.config, args.preset, args.output_root)
    print(f"Dataset written to {root}")


if __name__ == "__main__":
    main()
