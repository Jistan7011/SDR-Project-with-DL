from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.signal.oshea2018_waveform import (
    apply_channel_impairments,
    generate_clean_modulation,
    random_bits,
    to_unit_variance_iq,
)


def split_for_index(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cut = int(round(total * train_ratio))
    val_cut = train_cut + int(round(total * val_ratio))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def build_impairment_schedule(
    rng: np.random.Generator,
    count: int,
    snr_values: list[object],
    cfo_max: float,
    clock_sigma: float,
    gain_range: tuple[float, float],
    rolloff_range: tuple[float, float],
    multipath_spreads: list[object],
) -> list[dict[str, float]]:
    """Create a class-independent impairment schedule.

    The same index uses the same channel conditions for BASK, BFSK, and BPSK.
    This prevents the model from learning class-specific SNR/CFO/gain artifacts.
    """
    gain_min, gain_max = gain_range
    rolloff_min, rolloff_max = rolloff_range
    schedule: list[dict[str, float]] = []
    for _ in range(count):
        schedule.append(
            {
                "snr_db": float(rng.choice(snr_values)),
                "cfo_hz": float(rng.uniform(-cfo_max, cfo_max)),
                "clock_offset": float(rng.normal(0.0, clock_sigma)),
                "phase_offset": float(rng.uniform(0.0, 2.0 * np.pi)),
                "gain": float(rng.uniform(gain_min, gain_max)),
                "rolloff": float(rng.uniform(rolloff_min, rolloff_max)),
                "multipath_delay_spread": float(rng.choice(multipath_spreads)),
            }
        )
    return schedule


def generate_oshea2018_synthetic(config_path: str, preset: str = "smoke", output_root: str | None = None) -> Path:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root = ensure_dir(output_root or cfg["synthetic"]["root"])
    for split in ("train", "val", "test"):
        ensure_dir(root / split)

    sample_rate = float(cfg["sdr"]["sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    window_len = int(cfg["dataset"]["window_len"])
    samples_per_class = int(cfg["synthetic"]["samples_per_class"][preset])
    bit_count = int(cfg["synthetic"]["bits_per_example"])
    train_ratio = float(cfg["synthetic"]["train_ratio"])
    val_ratio = float(cfg["synthetic"]["val_ratio"])
    snr_values = list(cfg["synthetic"]["snr_db"])
    clock_sigma = float(cfg["synthetic"]["clock_offset_sigma"])
    cfo_max = float(cfg["synthetic"]["cfo_hz_max"])
    multipath_taps = int(cfg["synthetic"]["multipath_taps"])
    multipath_spreads = list(cfg["synthetic"]["multipath_delay_spread"])
    rolloff_min, rolloff_max = [float(x) for x in cfg["synthetic"]["rrc_rolloff_range"]]
    gain_min, gain_max = [float(x) for x in cfg["synthetic"]["gain_range"]]
    bfsk_freq_dev_hz = float(cfg["modulation"]["bfsk_freq_dev_hz"])
    impairment_schedule = build_impairment_schedule(
        rng,
        samples_per_class,
        snr_values,
        cfo_max,
        clock_sigma,
        (gain_min, gain_max),
        (rolloff_min, rolloff_max),
        multipath_spreads,
    )

    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    for modulation in CLASS_NAMES:
        for idx in tqdm(range(samples_per_class), desc=f"synthetic:{preset}:{modulation}"):
            split = split_for_index(idx, samples_per_class, train_ratio, val_ratio)
            bits = random_bits(rng, bit_count)
            impairment = impairment_schedule[idx]
            clean = generate_clean_modulation(
                modulation,
                bits,
                sample_rate=sample_rate,
                symbol_rate=symbol_rate,
                rolloff=impairment["rolloff"],
                bfsk_freq_dev_hz=bfsk_freq_dev_hz,
            )
            impaired = apply_channel_impairments(
                clean,
                rng,
                sample_rate=sample_rate,
                snr_db=impairment["snr_db"],
                cfo_hz=impairment["cfo_hz"],
                clock_offset=impairment["clock_offset"],
                phase_offset=impairment["phase_offset"],
                gain=impairment["gain"],
                multipath_taps=multipath_taps,
                multipath_delay_spread=impairment["multipath_delay_spread"],
            )
            channels = to_unit_variance_iq(impaired, window_len)
            name = f"sample_{counters[split]:07d}.npz"
            path = root / split / name
            np.savez(
                path,
                iq=channels,
                raw_iq=impaired[:window_len].astype(np.complex64),
                modulation=modulation,
                bits=bits,
                random_payload_bits=True,
                sample_rate=sample_rate,
                symbol_rate=symbol_rate,
                snr_db=impairment["snr_db"],
                cfo_hz=impairment["cfo_hz"],
                clock_offset=impairment["clock_offset"],
                phase_offset=impairment["phase_offset"],
                gain=impairment["gain"],
                rrc_rolloff=impairment["rolloff"],
                multipath_delay_spread=impairment["multipath_delay_spread"],
                dataset_kind="synthetic",
                experiment="Oshea2018",
            )
            manifest.append({"path": str(path), "split": split, "modulation": modulation, "snr_db": impairment["snr_db"]})
            counters[split] += 1
    write_json(root / f"manifest_{preset}.json", {"preset": preset, "counts": counters, "samples": manifest})
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--preset", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    root = generate_oshea2018_synthetic(args.config, args.preset, args.output_root)
    print(f"Synthetic Oshea2018 dataset written to {root}")


if __name__ == "__main__":
    main()
