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


def import_exp03_sessions(
    exp2_raw_root: str = "../../exp2/data/raw_iq",
    exp3_raw_root: str = "../data/raw_iq",
    output_root: str = "../data/processed",
    config_path: str = "../config/config.exp03.yaml",
    include_exp2_reference: bool = True,
    active_start_seconds: float = 1.1,
    active_duration_seconds: float = 4.5,
    windows_per_capture: int = 80,
    channel_bandwidth_hz: float = 100_000.0,
    target_sample_rate: float = 160_000.0,
    seed: int = 42,
    min_new_sessions: int = 6,
) -> Path:
    cfg = load_config(config_path)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(output_root)
    for split in ("train", "val", "test", "test_a", "test_b"):
        ensure_dir(root / split)

    session_dirs: list[Path] = []
    if include_exp2_reference:
        session_dirs.extend(find_sessions(Path(exp2_raw_root)))
    session_dirs.extend(find_sessions(Path(exp3_raw_root)))
    session_dirs = sorted({path.resolve(): path for path in session_dirs}.values(), key=lambda p: p.name)
    new_sessions = [path for path in session_dirs if session_number(path.name) >= 16]
    if len(new_sessions) < min_new_sessions:
        raise SystemExit(f"Need at least {min_new_sessions} Experiment 3 sessions >= session_016; found {len(new_sessions)}")

    sample_rate_default = float(cfg["sdr"]["rx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    window_size = int(cfg["dataset"]["window_size"])
    counters = {"train": 0, "val": 0, "test": 0, "test_a": 0, "test_b": 0}
    manifest: list[dict[str, object]] = []
    session_reports: list[dict[str, object]] = []

    for session_dir in session_dirs:
        metadata = read_json(session_dir / "metadata.json")
        session_id = str(metadata.get("session_id", session_dir.name))
        split = exp03_split(session_id)
        rx_sample_rate = float(metadata.get("rx_sample_rate", sample_rate_default))
        baseband_offset_hz = float(metadata.get("baseband_offset_hz", cfg.get("experiment3", {}).get("baseband_offset_hz", 500_000)))
        noise_path = resolve_source_path(metadata.get("noise_only_file", session_dir / "noise_only.bin"), session_dir)
        noise_iq = read_complex64(noise_path)
        noise_summary = spectrum_summary(noise_iq, rx_sample_rate)
        captures = list(metadata.get("captures", []))
        session_report = {"session_id": session_id, "split": split, "noise": noise_summary, "captures": []}

        for capture in captures:
            source = resolve_source_path(capture["file"], session_dir)
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
            offsets = rng.integers(0, len(active) - window_size + 1, size=windows_per_capture)
            bits = make_frame(payload)
            stem = f"{session_id}_{modulation.lower()}_{safe_payload(payload)}"
            for index, offset in enumerate(tqdm(offsets, desc=f"exp03:{stem}")):
                window = active[int(offset) : int(offset) + window_size]
                channels = iq_to_channels(window, window_size)
                save_sample(
                    root / split / f"{stem}_{index:06d}.npz",
                    channels,
                    window,
                    bits,
                    metadata,
                    noise_summary,
                    snr,
                    source,
                    active_start,
                    offset,
                    rx_sample_rate,
                    effective_rate,
                    modulation,
                    payload,
                    session_id,
                    baseband_offset_hz,
                    channel_bandwidth_hz,
                    symbol_rate,
                )
                manifest.append(sample_manifest(root / split / f"{stem}_{index:06d}.npz", split, session_id, modulation, payload, snr))
                counters[split] += 1
                if split == "test_b":
                    mirror = root / "test" / f"{stem}_{index:06d}.npz"
                    save_sample(
                        mirror,
                        channels,
                        window,
                        bits,
                        metadata,
                        noise_summary,
                        snr,
                        source,
                        active_start,
                        offset,
                        rx_sample_rate,
                        effective_rate,
                        modulation,
                        payload,
                        session_id,
                        baseband_offset_hz,
                        channel_bandwidth_hz,
                        symbol_rate,
                    )
                    manifest.append(sample_manifest(mirror, "test", session_id, modulation, payload, snr))
                    counters["test"] += 1
            session_report["captures"].append({"file": str(source), "modulation": modulation, "payload": payload, **snr})
        session_reports.append(session_report)

    validate_exp03_manifest(manifest)
    write_json(root / "manifest_exp03.json", {"counts": counters, "samples": manifest})
    write_json(root / "session_quality_report.json", {"sessions": session_reports})
    print(f"Imported Experiment 3 sessions to {root}: {counters}")
    return root


def save_sample(
    out_path: Path,
    channels: np.ndarray,
    window: np.ndarray,
    bits: np.ndarray,
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
        iq=channels,
        raw_iq=window.astype(np.complex64),
        modulation=modulation,
        payload=payload,
        bits=bits.astype(np.uint8),
        snr_db=float(snr["estimated_snr_db"]),
        estimated_snr_db=float(snr["estimated_snr_db"]),
        sample_rate=float(effective_rate),
        raw_sample_rate=rx_sample_rate,
        symbol_rate=symbol_rate,
        center_freq=float(metadata.get("center_freq", 433000000.0)),
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
        rf_path=str(metadata.get("rf_path", "ota_antenna")),
        rf_cable_between_sdr=bool(metadata.get("rf_cable_between_sdr", False)),
        tx_usb_connected_to_pc=bool(metadata.get("tx_usb_connected_to_pc", True)),
        rx_usb_connected_to_pc=bool(metadata.get("rx_usb_connected_to_pc", True)),
        tx_rx_distance_m=float(metadata.get("tx_rx_distance_m", np.nan)),
        antenna_layout=str(metadata.get("antenna_layout", "")),
        tx_antenna_orientation=str(metadata.get("tx_antenna_orientation", "")),
        rx_antenna_orientation=str(metadata.get("rx_antenna_orientation", "")),
        tx_height_cm=float_or_nan(metadata.get("tx_height_cm", np.nan)),
        rx_height_cm=float_or_nan(metadata.get("rx_height_cm", np.nan)),
        line_of_sight=bool(metadata.get("line_of_sight", True)),
        near_metal_objects=str(metadata.get("near_metal_objects", "")),
        human_nearby=str(metadata.get("human_nearby", "")),
    )


def find_sessions(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("session_*") if path.is_dir() and re.fullmatch(r"session_\d+", path.name))


def exp03_split(session_id: str) -> str:
    number = session_number(session_id)
    if 1 <= number <= 9:
        return "train"
    if 10 <= number <= 12:
        return "val"
    if 13 <= number <= 15:
        return "test_a"
    if number >= 16:
        return "test_b"
    raise ValueError(f"Unsupported session id for Experiment 3: {session_id}")


def session_number(session_id: str) -> int:
    return int(session_id.split("_")[-1])


def validate_exp03_manifest(manifest: list[dict[str, object]]) -> None:
    seen: dict[str, set[str]] = {}
    for row in manifest:
        split = str(row["split"])
        if split == "test":
            continue
        seen.setdefault(str(row["session_id"]), set()).add(split)
    leaked = {session: splits for session, splits in seen.items() if len(splits) > 1}
    if leaked:
        raise SystemExit(f"Session split leakage detected: {leaked}")
    test_b = {str(row["session_id"]) for row in manifest if row["split"] == "test_b"}
    if any(session_number(session) < 16 for session in test_b):
        raise SystemExit(f"test_b contains non-exp3 sessions: {sorted(test_b)}")


def resolve_source_path(value: object, session_dir: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    candidate = session_dir / path.name
    return candidate


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_complex64(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing IQ file: {path}")
    return np.fromfile(path, dtype=np.complex64)


def sample_manifest(path: Path, split: str, session_id: str, modulation: str, payload: str, snr: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "split": split, "session_id": session_id, "modulation": modulation, "payload": payload, "estimated_snr_db": float(snr["estimated_snr_db"])}


def safe_payload(payload: str) -> str:
    return "".join(ch if ch.isalnum() else f"x{ord(ch):02x}" for ch in payload)


def float_or_nan(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2-raw-root", default="../../exp2/data/raw_iq")
    parser.add_argument("--exp3-raw-root", default="../data/raw_iq")
    parser.add_argument("--output-root", default="../data/processed")
    parser.add_argument("--config", default="../config/config.exp03.yaml")
    parser.add_argument("--include-exp2-reference", action="store_true")
    parser.add_argument("--active-start-seconds", type=float, default=1.1)
    parser.add_argument("--active-duration-seconds", type=float, default=4.5)
    parser.add_argument("--windows-per-capture", type=int, default=80)
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100_000.0)
    parser.add_argument("--target-sample-rate", type=float, default=160_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-new-sessions", type=int, default=6)
    args = parser.parse_args()
    import_exp03_sessions(
        exp2_raw_root=args.exp2_raw_root,
        exp3_raw_root=args.exp3_raw_root,
        output_root=args.output_root,
        config_path=args.config,
        include_exp2_reference=args.include_exp2_reference,
        active_start_seconds=args.active_start_seconds,
        active_duration_seconds=args.active_duration_seconds,
        windows_per_capture=args.windows_per_capture,
        channel_bandwidth_hz=args.channel_bandwidth_hz,
        target_sample_rate=args.target_sample_rate,
        seed=args.seed,
        min_new_sessions=args.min_new_sessions,
    )


if __name__ == "__main__":
    main()
