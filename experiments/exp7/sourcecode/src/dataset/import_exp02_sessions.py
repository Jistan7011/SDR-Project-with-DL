from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, set_seed, write_json
from src.signal.channelize import channelize_and_downsample, estimate_snr_db, spectrum_summary
from src.signal.frame import make_frame
from src.signal.processing import iq_to_channels


def import_exp02_sessions(
    raw_root: str = "../data/raw_iq",
    output_root: str = "../data/processed",
    config_path: str = "../config/config.exp02.yaml",
    active_start_seconds: float = 1.1,
    active_duration_seconds: float = 4.5,
    windows_per_capture: int = 80,
    channel_bandwidth_hz: float = 100_000.0,
    target_sample_rate: float = 160_000.0,
    seed: int = 42,
    min_sessions: int = 12,
) -> Path:
    cfg = load_config(config_path)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(output_root)
    for split in ("train", "val", "test"):
        ensure_dir(root / split)
    sessions = sorted(
        path
        for path in Path(raw_root).glob("session_*")
        if path.is_dir() and re.fullmatch(r"session_\d+", path.name)
    )
    if len(sessions) < min_sessions:
        raise SystemExit(f"Need at least {min_sessions} session directories under {raw_root}; found {len(sessions)}")
    split_map = assign_session_splits([path.name for path in sessions])
    sample_rate_default = float(cfg["sdr"]["rx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    window_size = int(cfg["dataset"]["window_size"])
    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    session_reports: list[dict[str, object]] = []

    for session_dir in sessions:
        metadata = read_json(session_dir / "metadata.json")
        session_id = str(metadata.get("session_id", session_dir.name))
        split = split_map[session_id]
        rx_sample_rate = float(metadata.get("rx_sample_rate", sample_rate_default))
        baseband_offset_hz = float(metadata.get("baseband_offset_hz", cfg.get("experiment2", {}).get("baseband_offset_hz", 250_000)))
        noise_path = Path(str(metadata.get("noise_only_file", session_dir / "noise_only.bin")))
        noise_iq = read_complex64(noise_path)
        noise_summary = spectrum_summary(noise_iq, rx_sample_rate)
        captures = list(metadata.get("captures", []))
        session_report = {"session_id": session_id, "split": split, "noise": noise_summary, "captures": []}

        for capture in captures:
            source = Path(str(capture["file"]))
            modulation = str(capture["modulation"]).upper()
            payload = str(capture["payload"])
            if modulation not in CLASS_NAMES:
                raise SystemExit(f"Unsupported modulation in {source}: {modulation}")
            raw_iq = read_complex64(source)
            active_start = int(active_start_seconds * rx_sample_rate)
            active_end = min(len(raw_iq), active_start + int(active_duration_seconds * rx_sample_rate))
            active_raw = raw_iq[active_start:active_end]
            if len(active_raw) < window_size:
                raise SystemExit(f"Active segment too short in {source}: {len(active_raw)} raw samples")
            snr = estimate_snr_db(noise_iq, active_raw)
            active, effective_rate = channelize_and_downsample(
                active_raw,
                sample_rate=rx_sample_rate,
                channel_center_hz=baseband_offset_hz,
                channel_bandwidth_hz=channel_bandwidth_hz,
                target_sample_rate=target_sample_rate,
            )
            if len(active) < window_size:
                raise SystemExit(f"Post-channelized segment too short in {source}: {len(active)} samples")
            max_offset = len(active) - window_size
            offsets = rng.integers(0, max_offset + 1, size=windows_per_capture)
            bits = make_frame(payload)
            stem = f"{session_id}_{modulation.lower()}_{safe_payload(payload)}"
            for index, offset in enumerate(tqdm(offsets, desc=f"exp02:{stem}")):
                window = active[int(offset) : int(offset) + window_size]
                out_path = root / split / f"{stem}_{index:06d}.npz"
                np.savez(
                    out_path,
                    iq=iq_to_channels(window, window_size),
                    raw_iq=window.astype(np.complex64),
                    modulation=modulation,
                    payload=payload,
                    bits=bits.astype(np.uint8),
                    snr_db=float(snr["estimated_snr_db"]),
                    estimated_snr_db=float(snr["estimated_snr_db"]),
                    sample_rate=float(effective_rate),
                    raw_sample_rate=rx_sample_rate,
                    symbol_rate=symbol_rate,
                    center_freq=float(metadata.get("center_freq", cfg["sdr"]["center_freq"])),
                    session_id=session_id,
                    source_file=str(source),
                    source_offset=int(active_start + round(float(offset) * rx_sample_rate / effective_rate)),
                    baseband_offset_hz=baseband_offset_hz,
                    channel_bandwidth_hz=channel_bandwidth_hz,
                    tx_vga_gain=float(metadata.get("tx_vga_gain", np.nan)),
                    tx_amp_gain=float(metadata.get("tx_amp_gain", np.nan)),
                    rx_gain=float(metadata.get("rx_gain", np.nan)),
                    noise_power=float(snr["noise_power"]),
                    active_power=float(snr["active_power"]),
                    signal_power=float(snr["signal_power"]),
                    noise_floor_db=float(noise_summary["noise_floor_db"]),
                    peak_frequency_hz=float(noise_summary["peak_frequency_hz"]),
                    interference_flag=bool(noise_summary["interference_flag"]),
                )
                manifest.append(
                    {
                        "path": str(out_path),
                        "split": split,
                        "session_id": session_id,
                        "modulation": modulation,
                        "payload": payload,
                        "estimated_snr_db": float(snr["estimated_snr_db"]),
                    }
                )
                counters[split] += 1
            session_report["captures"].append({"file": str(source), "modulation": modulation, "payload": payload, **snr})
        session_reports.append(session_report)

    validate_session_split(manifest)
    write_json(root / "manifest_exp02.json", {"counts": counters, "session_split": split_map, "samples": manifest})
    write_json(root / "session_quality_report.json", {"sessions": session_reports})
    print(f"Imported Experiment 2 sessions to {root}: {counters}")
    return root


def assign_session_splits(session_ids: list[str]) -> dict[str, str]:
    total = len(session_ids)
    if total >= 15:
        train_n, val_n = 9, 3
    elif total >= 12:
        train_n, val_n = 7, 2
    else:
        raise ValueError("Experiment 2 requires at least 12 sessions")
    mapping: dict[str, str] = {}
    for index, session_id in enumerate(session_ids):
        if index < train_n:
            split = "train"
        elif index < train_n + val_n:
            split = "val"
        else:
            split = "test"
        mapping[session_id] = split
    return mapping


def validate_session_split(manifest: list[dict[str, object]]) -> None:
    seen: dict[str, set[str]] = {}
    payloads_by_mod: dict[str, set[str]] = {name: set() for name in CLASS_NAMES}
    for row in manifest:
        seen.setdefault(str(row["session_id"]), set()).add(str(row["split"]))
        payloads_by_mod[str(row["modulation"])].add(str(row["payload"]))
    leaked = {session: splits for session, splits in seen.items() if len(splits) > 1}
    if leaked:
        raise SystemExit(f"Session split leakage detected: {leaked}")
    pools = list(payloads_by_mod.values())
    if pools and any(pool != pools[0] for pool in pools):
        raise SystemExit(f"Payload pools differ by modulation: {payloads_by_mod}")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_complex64(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing IQ file: {path}")
    return np.fromfile(path, dtype=np.complex64)


def safe_payload(payload: str) -> str:
    return "".join(ch if ch.isalnum() else f"x{ord(ch):02x}" for ch in payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="../data/raw_iq")
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--config", default="../config/config.exp02.yaml")
    parser.add_argument("--active-start-seconds", type=float, default=1.1)
    parser.add_argument("--active-duration-seconds", type=float, default=4.5)
    parser.add_argument("--windows-per-capture", type=int, default=80)
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100_000.0)
    parser.add_argument("--target-sample-rate", type=float, default=160_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sessions", type=int, default=12)
    args = parser.parse_args()
    import_exp02_sessions(
        raw_root=args.raw_root,
        output_root=args.output_root,
        config_path=args.config,
        active_start_seconds=args.active_start_seconds,
        active_duration_seconds=args.active_duration_seconds,
        windows_per_capture=args.windows_per_capture,
        channel_bandwidth_hz=args.channel_bandwidth_hz,
        target_sample_rate=args.target_sample_rate,
        seed=args.seed,
        min_sessions=args.min_sessions,
    )


if __name__ == "__main__":
    main()
