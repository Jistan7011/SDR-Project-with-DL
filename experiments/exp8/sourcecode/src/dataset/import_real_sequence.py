from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter
from tqdm import tqdm

from src.common import ensure_dir, load_config, set_seed, write_json
from src.signal.frame import make_frame
from src.signal.processing import iq_to_channels


SEQUENCE = [
    ("bask_a.bin", "BASK", "A"),
    ("bfsk_f.bin", "BFSK", "F"),
    ("bpsk_p.bin", "BPSK", "P"),
]


def import_real_sequence(
    raw_dir: str = "data/real/raw_iq",
    output_root: str = "data/real/processed",
    config_path: str = "config.yaml",
    active_start_seconds: float = 1.1,
    active_duration_seconds: float = 2.6,
    windows_per_class: int = 900,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    channel_center_hz: float | None = None,
    channel_bandwidth_hz: float | None = None,
    seed: int = 42,
) -> Path:
    cfg = load_config(config_path)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    window_size = int(cfg["dataset"]["window_size"])
    root = Path(output_root)
    for split in ("train", "val", "test"):
        ensure_dir(root / split)

    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    for filename, modulation, payload in SEQUENCE:
        source = Path(raw_dir) / filename
        iq = np.fromfile(source, dtype=np.complex64)
        start = int(active_start_seconds * sample_rate)
        end = min(len(iq), start + int(active_duration_seconds * sample_rate))
        active = iq[start:end]
        if len(active) < window_size:
            raise SystemExit(f"{source} active segment is too short: {len(active)} samples")
        if channel_center_hz is not None and channel_bandwidth_hz is not None:
            active = channel_filter(active, sample_rate, channel_center_hz, channel_bandwidth_hz, target_center_hz=100_000.0)

        max_offset = len(active) - window_size
        offsets = rng.integers(0, max_offset + 1, size=windows_per_class)
        split_names = make_splits(windows_per_class, train_ratio, val_ratio)
        bits = make_frame(payload)
        for idx, (offset, split) in enumerate(tqdm(list(zip(offsets, split_names)), desc=f"import-seq:{source.stem}")):
            window = active[int(offset) : int(offset) + window_size]
            out_path = root / split / f"{source.stem}_{idx:06d}.npz"
            np.savez(
                out_path,
                iq=iq_to_channels(window, window_size),
                raw_iq=window.astype(np.complex64),
                modulation=modulation,
                payload=payload,
                bits=bits.astype(np.uint8),
                snr_db=np.nan,
                sample_rate=sample_rate,
                symbol_rate=symbol_rate,
                center_freq=int(cfg["sdr"]["center_freq"]),
                session_id=source.stem,
                source_file=str(source),
                source_offset=int(start + offset),
            )
            manifest.append({"path": str(out_path), "split": split, "modulation": modulation, "payload": payload})
            counters[split] += 1

    write_json(root / "manifest_real_sequence.json", {"counts": counters, "samples": manifest})
    print(f"Imported real sequence to {root}: {counters}")
    return root


def make_splits(n: int, train_ratio: float, val_ratio: float) -> list[str]:
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    return ["train"] * train_n + ["val"] * val_n + ["test"] * test_n


def channel_filter(iq: np.ndarray, sample_rate: float, channel_center_hz: float, channel_bandwidth_hz: float, target_center_hz: float) -> np.ndarray:
    t = np.arange(len(iq), dtype=np.float64) / sample_rate
    shifted = iq * np.exp(-1j * 2.0 * np.pi * channel_center_hz * t)
    fir = firwin(129, cutoff=min(channel_bandwidth_hz / 2.0, sample_rate * 0.45), fs=sample_rate)
    filtered = lfilter(fir, [1.0], shifted)
    return (filtered * np.exp(1j * 2.0 * np.pi * target_center_hz * t)).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/real/raw_iq")
    parser.add_argument("--output-root", default="data/real/processed")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--active-start-seconds", type=float, default=1.1)
    parser.add_argument("--active-duration-seconds", type=float, default=2.6)
    parser.add_argument("--windows-per-class", type=int, default=900)
    parser.add_argument("--channel-center-hz", type=float, default=None)
    parser.add_argument("--channel-bandwidth-hz", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import_real_sequence(
        raw_dir=args.raw_dir,
        output_root=args.output_root,
        config_path=args.config,
        active_start_seconds=args.active_start_seconds,
        active_duration_seconds=args.active_duration_seconds,
        windows_per_class=args.windows_per_class,
        channel_center_hz=args.channel_center_hz,
        channel_bandwidth_hz=args.channel_bandwidth_hz,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
