from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.dataset.split_by_session import split_name_for_class
from src.signal.awgn import add_awgn
from src.signal.channel import apply_channel_impairments, params_from_config
from src.signal.frame import make_frame, make_frame_from_bits
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
    modulation_cfg = cfg["modulation"]
    payloads = list(modulation_cfg.get("payloads", ["A", "F", "P"]))
    payload_mode = str(modulation_cfg.get("payload_mode", "random_bytes"))
    payload_bytes = int(modulation_cfg.get("payload_bytes", 16))
    snrs = list(cfg["awgn"]["snr_list"])
    channel_params = params_from_config(cfg)
    samples_per_class = int(cfg["dataset"]["samples_per_class"][preset])
    total_sessions = samples_per_class * len(CLASS_NAMES)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    session = 0
    for modulation in CLASS_NAMES:
        for class_sample in tqdm(range(samples_per_class), desc=f"{preset}:{modulation}"):
            payload, payload_bits, payload_hex = generate_payload(payload_mode, payloads, payload_bytes, rng)
            snr_db = float(rng.choice(snrs))
            bits = make_frame_from_bits(payload_bits) if payload_mode == "random_bytes" else make_frame(payload)
            mod_params = random_modulation_params(modulation, modulation_cfg, rng)
            clean = modulate_bits(modulation, bits, sps, sample_rate, mod_params)
            impaired, impairment_meta = apply_channel_impairments(clean, sample_rate, rng, channel_params)
            noisy = add_awgn(impaired, snr_db, rng)
            window, window_offset = random_window(noisy, window_size, rng)
            channels = iq_to_channels(window, window_size)
            split = split_name_for_class(class_sample, samples_per_class, float(cfg["dataset"]["train_ratio"]), float(cfg["dataset"]["val_ratio"]))
            name = f"sample_{counters[split]:06d}.npz"
            path = root / split / name
            np.savez(
                path,
                iq=channels,
                raw_iq=window.astype(np.complex64),
                modulation=modulation,
                payload=payload,
                payload_hex=payload_hex,
                payload_mode=payload_mode,
                payload_bytes=int(payload_bytes),
                bits=bits.astype(np.uint8),
                snr_db=snr_db,
                sample_rate=sample_rate,
                symbol_rate=symbol_rate,
                center_freq=int(cfg["sdr"]["center_freq"]),
                session_id=f"session_{session:06d}",
                source_offset=int(window_offset),
                frame_samples=int(len(noisy)),
                modulation_params=mod_params,
                impairment=impairment_meta,
            )
            manifest.append(
                {
                    "path": str(path),
                    "split": split,
                    "modulation": modulation,
                    "payload": payload,
                    "payload_hex": payload_hex,
                    "payload_mode": payload_mode,
                    "payload_bytes": int(payload_bytes),
                    "snr_db": snr_db,
                    "source_offset": int(window_offset),
                    "modulation_params": mod_params,
                    "impairment": impairment_meta,
                }
            )
            counters[split] += 1
            session += 1

    write_json(root / f"manifest_{preset}.json", {"preset": preset, "counts": counters, "samples": manifest})
    return root


def generate_payload(
    payload_mode: str,
    payloads: list[object],
    payload_bytes: int,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray, str]:
    if payload_mode == "random_bytes":
        values = rng.integers(0, 256, size=max(1, payload_bytes), dtype=np.uint8)
        bits = np.unpackbits(values)
        return values.tobytes().hex(), bits.astype(np.uint8), values.tobytes().hex()
    if payload_mode == "random_text":
        alphabet = [str(item) for item in payloads] or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        payload = "".join(str(rng.choice(alphabet)) for _ in range(max(1, payload_bytes)))
        return payload, np.asarray([], dtype=np.uint8), payload.encode("utf-8", errors="replace").hex()
    payload = str(rng.choice(payloads))
    return payload, np.asarray([], dtype=np.uint8), payload.encode("utf-8", errors="replace").hex()


def random_window(iq: np.ndarray, window_size: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if len(iq) <= window_size:
        return iq, 0
    offset = int(rng.integers(0, len(iq) - window_size + 1))
    return iq[offset : offset + window_size], offset


def random_modulation_params(
    modulation: str,
    modulation_cfg: dict[str, object],
    rng: np.random.Generator,
) -> dict[str, float]:
    randomize = modulation_cfg.get("randomize", {}) or {}
    mod = modulation.upper()
    if mod == "BASK":
        amp0 = draw_range(randomize.get("bask_amp0", [0.0, 0.35]), rng)
        amp1 = draw_range(randomize.get("bask_amp1", [0.65, 1.0]), rng)
        if amp1 <= amp0:
            amp0, amp1 = min(amp0, amp1), max(amp0, amp1) + 1e-3
        return {
            "carrier_freq": draw_range(randomize.get("bask_carrier_hz", [70_000.0, 130_000.0]), rng),
            "amp0": amp0,
            "amp1": amp1,
        }
    if mod == "BFSK":
        center = draw_range(randomize.get("bfsk_center_hz", [80_000.0, 120_000.0]), rng)
        spacing = draw_range(randomize.get("bfsk_spacing_hz", [20_000.0, 70_000.0]), rng)
        return {
            "f0": max(1_000.0, center - spacing / 2.0),
            "f1": center + spacing / 2.0,
        }
    if mod == "BPSK":
        phase_error = draw_range(randomize.get("bpsk_phase_error_rad", [-0.25, 0.25]), rng)
        return {
            "carrier_freq": draw_range(randomize.get("bpsk_carrier_hz", [70_000.0, 130_000.0]), rng),
            "phase0": 0.0,
            "phase1": float(np.pi + phase_error),
        }
    return {}


def draw_range(value: object, rng: np.random.Generator) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    bounds = list(value)  # type: ignore[arg-type]
    if len(bounds) != 2:
        raise ValueError(f"Expected two-value range, got {value}")
    lo, hi = float(bounds[0]), float(bounds[1])
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


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
