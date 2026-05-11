from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.signal.channelize import channelize_and_downsample, estimate_snr_db, spectrum_summary
from src.signal.processing import iq_to_channels


def import_exp05_classification_windows(
    raw_roots: list[str],
    output_root: str = "../data/processed",
    config_path: str = "../config/config.exp05.yaml",
    active_start_seconds: float | None = None,
    active_duration_seconds: float | None = None,
    windows_per_capture: int | None = None,
    target_sample_rate: float | None = None,
    channel_bandwidth_hz: float | None = None,
) -> Path:
    cfg = load_config(config_path)
    exp_cfg = cfg.get("experiment5", {})
    out_root = ensure_dir(output_root)
    for split in ("train", "val", "test"):
        ensure_dir(out_root / split)
    target_rate = float(target_sample_rate or cfg["dataset"]["target_sample_rate"])
    bandwidth = float(channel_bandwidth_hz or cfg["dataset"]["channel_bandwidth_hz"])
    active_start = float(active_start_seconds if active_start_seconds is not None else exp_cfg.get("active_start_seconds", 1.1))
    active_duration = float(active_duration_seconds if active_duration_seconds is not None else exp_cfg.get("active_duration_seconds", 5.0))
    count = int(windows_per_capture or exp_cfg.get("classification_windows_per_capture", 20))
    window_size = int(cfg["dataset"]["window_size"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    session_dirs = find_session_dirs(raw_roots)
    minimum_sessions = int(exp_cfg.get("minimum_sessions", 1))
    if len(session_dirs) < minimum_sessions:
        raise SystemExit(f"Experiment 5 requires at least {minimum_sessions} sessions; found {len(session_dirs)}")

    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    sample_index = 0
    for session_dir in session_dirs:
        metadata = read_json(session_dir / "metadata.json")
        session_id = str(metadata.get("session_id", session_dir.name))
        split = split_for_session(session_id, cfg)
        rx_sample_rate = float(metadata.get("rx_sample_rate", cfg["sdr"]["rx_sample_rate"]))
        baseband_offset = float(metadata.get("baseband_offset_hz", cfg["sdr"].get("baseband_offset_hz", 500_000)))
        noise_path = resolve_path(metadata.get("noise_only_file", session_dir / "noise_only.bin"), session_dir)
        noise_iq = read_complex64(noise_path)
        noise_summary = spectrum_summary(noise_iq, rx_sample_rate)
        for capture in tqdm(list(metadata.get("captures", [])), desc=f"exp05:{session_id}"):
            source = resolve_path(capture["file"], session_dir)
            modulation = str(capture["modulation"]).upper()
            payload = str(capture["payload"])
            if modulation not in CLASS_NAMES:
                raise SystemExit(f"Unsupported modulation in {source}: {modulation}")
            raw_iq = read_complex64(source)
            active_start_index = int(active_start * rx_sample_rate)
            active_end = min(len(raw_iq), active_start_index + int(active_duration * rx_sample_rate))
            active_raw = raw_iq[active_start_index:active_end]
            if len(active_raw) == 0:
                raise SystemExit(f"Active segment is empty: {source}")
            snr = estimate_snr_db(noise_iq, active_raw)
            active, effective_rate = channelize_and_downsample(
                active_raw,
                sample_rate=rx_sample_rate,
                channel_center_hz=baseband_offset,
                channel_bandwidth_hz=bandwidth,
                target_sample_rate=target_rate,
            )
            if len(active) < window_size:
                raise SystemExit(f"Channelized segment too short in {source}: {len(active)}")
            for local_index, offset in enumerate(window_offsets(len(active), window_size, count)):
                channels = iq_to_channels(active[offset : offset + window_size], window_size)
                out_path = out_root / split / f"{session_id}_{modulation.lower()}_{safe_payload(payload)}_{local_index:03d}.npz"
                save_sample(
                    out_path,
                    channels,
                    metadata,
                    noise_summary,
                    snr,
                    source,
                    active_start_index,
                    offset,
                    rx_sample_rate,
                    effective_rate,
                    modulation,
                    payload,
                    session_id,
                    baseband_offset,
                    bandwidth,
                    symbol_rate,
                )
                manifest.append({"path": str(out_path), "split": split, "session_id": session_id, "modulation": modulation, "payload": payload})
                sample_index += 1
                counters[split] += 1
    validate_no_session_leakage(manifest)
    write_json(
        out_root / "manifest_exp05_classification.json",
        {"sample_count": sample_index, "counts": counters, "raw_roots": raw_roots, "samples": manifest},
    )
    print(f"Imported {sample_index} classification samples to {out_root}: {counters}")
    return out_root


def find_session_dirs(raw_roots: list[str]) -> list[Path]:
    sessions: dict[str, Path] = {}
    for raw_root in raw_roots:
        for path in sorted(Path(raw_root).glob("session_*")):
            if path.is_dir() and (path / "metadata.json").exists():
                sessions.setdefault(path.name, path)
    return [sessions[key] for key in sorted(sessions)]


def window_offsets(length: int, window_size: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    max_start = max(0, length - window_size)
    return [int(round(value)) for value in np.linspace(0, max_start, count)]


def save_sample(
    out_path: Path,
    channels: np.ndarray,
    metadata: dict[str, object],
    noise_summary: dict[str, object],
    snr: dict[str, object],
    source: Path,
    active_start: int,
    offset: int,
    rx_sample_rate: float,
    effective_rate: float,
    modulation: str,
    payload: str,
    session_id: str,
    baseband_offset_hz: float,
    channel_bandwidth_hz: float,
    symbol_rate: float,
) -> None:
    np.savez(
        out_path,
        iq=channels.astype(np.float32),
        modulation=modulation,
        payload=payload,
        sample_rate=float(effective_rate),
        raw_sample_rate=float(rx_sample_rate),
        symbol_rate=float(symbol_rate),
        center_freq=float(metadata.get("center_freq", 433000000.0)),
        session_id=session_id,
        source_file=str(source),
        source_offset=int(active_start + round(float(offset) * rx_sample_rate / effective_rate)),
        baseband_offset_hz=float(baseband_offset_hz),
        channel_bandwidth_hz=float(channel_bandwidth_hz),
        snr_db=float(snr["estimated_snr_db"]),
        estimated_snr_db=float(snr["estimated_snr_db"]),
        tx_vga_gain=float(metadata.get("tx_vga_gain", np.nan)),
        tx_amp_gain=float(metadata.get("tx_amp_gain", np.nan)),
        rx_gain=float(metadata.get("rx_gain", np.nan)),
        noise_power=float(snr["noise_power"]),
        active_power=float(snr["active_power"]),
        signal_power=float(snr["signal_power"]),
        noise_floor_db=float(noise_summary["noise_floor_db"]),
        peak_frequency_hz=float(noise_summary["peak_frequency_hz"]),
        interference_flag=bool(noise_summary["interference_flag"]),
        rf_path=str(metadata.get("rf_path", "ota_antenna")),
        rf_cable_between_sdr=bool(metadata.get("rf_cable_between_sdr", False)),
        tx_rx_distance_m=float(metadata.get("tx_rx_distance_m", np.nan)),
        antenna_layout=str(metadata.get("antenna_layout", "")),
        antenna_note=str(metadata.get("antenna_note", "")),
    )


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_complex64(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing IQ file: {path}")
    return np.fromfile(path, dtype=np.complex64)


def resolve_path(value: object, session_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.exists() else session_dir / path.name


def safe_payload(payload: str) -> str:
    return "".join(ch if ch.isalnum() else f"x{ord(ch):02x}" for ch in payload)


def split_for_session(session_id: str, cfg: dict[str, object]) -> str:
    policy = dict(cfg.get("experiment5", {}).get("split_policy", {}))
    for split, sessions in policy.items():
        if session_id in list(sessions):
            return str(split)
    return "test"


def validate_no_session_leakage(manifest: list[dict[str, object]]) -> None:
    seen: dict[str, set[str]] = {}
    for row in manifest:
        seen.setdefault(str(row["session_id"]), set()).add(str(row["split"]))
    leaked = {session: splits for session, splits in seen.items() if len(splits) > 1}
    if leaked:
        raise SystemExit(f"Session split leakage detected: {leaked}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-roots", nargs="+", default=["../../exp4/data/raw_iq", "../data/raw_iq"])
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--config", default="../config/config.exp05.yaml")
    parser.add_argument("--active-start-seconds", type=float, default=None)
    parser.add_argument("--active-duration-seconds", type=float, default=None)
    parser.add_argument("--windows-per-capture", type=int, default=None)
    parser.add_argument("--target-sample-rate", type=float, default=None)
    parser.add_argument("--channel-bandwidth-hz", type=float, default=None)
    args = parser.parse_args()
    import_exp05_classification_windows(
        raw_roots=args.raw_roots,
        output_root=args.output_root,
        config_path=args.config,
        active_start_seconds=args.active_start_seconds,
        active_duration_seconds=args.active_duration_seconds,
        windows_per_capture=args.windows_per_capture,
        target_sample_rate=args.target_sample_rate,
        channel_bandwidth_hz=args.channel_bandwidth_hz,
    )


if __name__ == "__main__":
    main()
