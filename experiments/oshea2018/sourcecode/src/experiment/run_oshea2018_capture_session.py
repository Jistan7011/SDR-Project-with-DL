from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.import_oshea2018_ota_windows import clipping_rate, spectral_peak_prominence, window_rms
from src.signal.channelize import channelize_and_downsample, estimate_snr_db


def radioconda_python_cmd(radioconda_root: str, python_args: list[str]) -> list[str]:
    activate = str(Path(radioconda_root) / "Scripts" / "activate.bat")
    return ["cmd", "/c", "call", activate, radioconda_root, "&&", "python", *python_args]


def stable_payload_seed(session_id: str, capture_idx: int) -> int:
    """Use the same payload seed across modulations for a session/capture index."""
    digest = hashlib.sha256(f"{session_id}:payload:{capture_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def resolve_session_settings(cfg: dict, session_id: str, args: dict[str, float | None]) -> dict[str, float]:
    session_num = int(session_id.split("_")[-1])
    settings = {
        "tx_vga_gain": float(cfg["sdr"].get("tx_vga_gain", 40.0)),
        "tx_amp_gain": float(cfg["sdr"].get("tx_amp_gain", 0.0)),
        "rx_gain": float(cfg["sdr"].get("rx_gain", 20.0)),
        "baseband_offset_hz": float(cfg["sdr"].get("baseband_offset_hz", 500_000.0)),
    }
    for item in cfg.get("ota", {}).get("session_plan", []):
        start, end = item.get("session_range", [0, -1])
        if int(start) <= session_num <= int(end):
            for key in settings:
                if key in item:
                    settings[key] = float(item[key])
    for key, value in args.items():
        if value is not None:
            settings[key] = float(value)
    return settings


def quality_against_noise(capture_path: Path, noise_path: Path, sample_rate: float, cfg: dict, baseband_offset_hz: float) -> dict[str, object]:
    capture = np.fromfile(capture_path, dtype=np.complex64)
    noise = np.fromfile(noise_path, dtype=np.complex64)
    active_start_seconds = float(cfg["ota"].get("active_start_seconds", 1.1))
    active_duration_seconds = float(cfg["ota"].get("active_duration_seconds", 3.8))
    start = min(len(capture), int(round(active_start_seconds * sample_rate)))
    end = min(len(capture), start + int(round(active_duration_seconds * sample_rate)))
    active = capture[start:end]
    target_sample_rate = float(cfg["dataset"].get("target_sample_rate", sample_rate))
    channel_bandwidth_hz = float(cfg["dataset"].get("channel_bandwidth_hz", 100_000.0))
    active, effective_sample_rate = channelize_and_downsample(
        active,
        sample_rate=sample_rate,
        channel_center_hz=baseband_offset_hz,
        channel_bandwidth_hz=channel_bandwidth_hz,
        target_sample_rate=target_sample_rate,
    )
    noise, _ = channelize_and_downsample(
        noise,
        sample_rate=sample_rate,
        channel_center_hz=baseband_offset_hz,
        channel_bandwidth_hz=channel_bandwidth_hz,
        target_sample_rate=target_sample_rate,
    )
    noise_rms = window_rms(noise)
    tx_rms = window_rms(active)
    ratio = float(tx_rms / max(noise_rms, 1e-12))
    peak = spectral_peak_prominence(active)
    clip = clipping_rate(active)
    snr_db = estimate_snr_db(noise, active)["estimated_snr_db"]
    qcfg = cfg["ota"].get("quality_filter", {})
    min_ratio = float(qcfg.get("min_tx_to_noise_rms_ratio", 2.0))
    min_snr_db = qcfg.get("min_estimated_snr_db", None)
    max_clip = float(qcfg.get("max_clipping_rate", 0.02))
    reasons: list[str] = []
    if ratio < min_ratio:
        reasons.append(f"low_tx_to_noise_rms_ratio:{ratio:.3f}<{min_ratio:.3f}")
    if min_snr_db is not None and snr_db < float(min_snr_db):
        reasons.append(f"low_estimated_snr_db:{snr_db:.2f}<{float(min_snr_db):.2f}")
    if clip > max_clip:
        reasons.append(f"high_clipping_rate:{clip:.3f}>{max_clip:.3f}")
    return {
        "noise_rms": noise_rms,
        "tx_rms": tx_rms,
        "tx_to_noise_rms_ratio": ratio,
        "estimated_snr_db": snr_db,
        "spectral_peak_prominence": peak,
        "clipping_rate": clip,
        "quality_pass": len(reasons) == 0,
        "quality_reason": "pass" if not reasons else ";".join(reasons),
        "active_start_seconds": active_start_seconds,
        "active_duration_seconds": active_duration_seconds,
        "quality_band": "channelized",
        "effective_sample_rate": effective_sample_rate,
    }


def wait_process(proc: subprocess.Popen, timeout_seconds: float, label: str) -> int:
    try:
        return int(proc.wait(timeout=max(1.0, timeout_seconds)))
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc)
        raise SystemExit(f"{label} timed out after {timeout_seconds:.1f}s")


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def run_oshea2018_capture_session(
    session_id: str,
    config_path: str = "../config/config.oshea2018.yaml",
    output_root: str = "../data/raw_ota_clean",
    radioconda_root: str | None = None,
    captures_per_class: int = 10,
    tx_vga_gain: float | None = None,
    tx_amp_gain: float | None = None,
    rx_gain: float | None = None,
    baseband_offset_hz: float | None = None,
    max_retries: int = 3,
    quality_gate: bool = True,
    dry_run: bool = False,
    process_timeout_margin_seconds: float = 60.0,
    tx_backend: str | None = None,
) -> Path:
    radioconda_root = radioconda_root or os.environ.get("RADIOCONDA_ROOT") or r"C:\Users\qus70\radioconda"
    cfg = load_config(config_path)
    session_dir = ensure_dir(Path(output_root) / session_id)
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    settings = resolve_session_settings(
        cfg,
        session_id,
        {
            "tx_vga_gain": tx_vga_gain,
            "tx_amp_gain": tx_amp_gain,
            "rx_gain": rx_gain,
            "baseband_offset_hz": baseband_offset_hz,
        },
    )
    capture_seconds = float(cfg["ota"]["capture_seconds"])
    noise_seconds = float(cfg["ota"].get("noise_capture_seconds", 2.0))
    tx_seconds = max(0.5, capture_seconds - float(cfg["ota"]["rx_lead_seconds"]))
    tx_backend_value = str(tx_backend or cfg["sdr"].get("tx_backend", "hackrf_transfer"))
    captures: list[dict[str, object]] = []

    noise_output = session_dir / "noise_only.bin"
    noise_cmd = radioconda_python_cmd(
        radioconda_root,
        [
            "-m",
            "src.sdr.capture_iq",
            "--config",
            config_path,
            "--output",
            str(noise_output),
            "--seconds",
            str(noise_seconds),
            "--sample-rate",
            str(sample_rate),
            "--rx-gain",
            str(settings["rx_gain"]),
        ],
    )
    print(f"\n{session_id}: noise_only")
    print("RX:", " ".join(noise_cmd))
    if not dry_run:
        noise_proc = subprocess.Popen(noise_cmd)
        noise_code = wait_process(noise_proc, noise_seconds + process_timeout_margin_seconds, "noise-only RX")
        if noise_code != 0:
            raise SystemExit(f"noise-only RX failed: {noise_code}")

    for capture_idx in range(captures_per_class):
        payload_seed = stable_payload_seed(session_id, capture_idx)
        for modulation in CLASS_NAMES:
            output = session_dir / f"{session_id}_{modulation.lower()}_{capture_idx:03d}.bin"
            quality: dict[str, object] = {"quality_pass": True, "quality_reason": "dry_run"}
            for attempt in range(max(1, max_retries)):
                rx_cmd = radioconda_python_cmd(
                    radioconda_root,
                    [
                        "-m",
                        "src.sdr.capture_iq",
                        "--config",
                        config_path,
                        "--output",
                        str(output),
                        "--seconds",
                        str(capture_seconds),
                        "--sample-rate",
                        str(sample_rate),
                        "--rx-gain",
                        str(settings["rx_gain"]),
                    ],
                )
                tx_cmd = radioconda_python_cmd(
                    radioconda_root,
                    [
                        "-m",
                        "src.sdr.hackrf_tx_oshea2018",
                        "--config",
                        config_path,
                        "--modulation",
                        modulation,
                        "--seconds",
                        str(tx_seconds),
                        "--seed",
                        str(payload_seed),
                        "--tx-vga-gain",
                        str(settings["tx_vga_gain"]),
                        "--tx-amp-gain",
                        str(settings["tx_amp_gain"]),
                        "--baseband-offset-hz",
                        str(settings["baseband_offset_hz"]),
                        "--backend",
                        tx_backend_value,
                    ],
                )
                print(f"\n{session_id}: {modulation} capture={capture_idx} attempt={attempt + 1} payload_seed={payload_seed}")
                print("RX:", " ".join(rx_cmd))
                print("TX:", " ".join(tx_cmd))
                if dry_run:
                    break
                rx_proc = subprocess.Popen(rx_cmd)
                time.sleep(float(cfg["ota"]["rx_lead_seconds"]))
                tx_proc = subprocess.Popen(tx_cmd)
                try:
                    tx_code = wait_process(tx_proc, tx_seconds + process_timeout_margin_seconds, "TX")
                    rx_code = wait_process(rx_proc, capture_seconds + process_timeout_margin_seconds, "RX")
                    if tx_code != 0:
                        raise SystemExit(f"TX failed: {tx_code}")
                    if rx_code != 0:
                        raise SystemExit(f"RX failed: {rx_code}")
                except BaseException:
                    for proc in (tx_proc, rx_proc):
                        if proc.poll() is None:
                            terminate_process_tree(proc)
                    raise
                quality = quality_against_noise(output, noise_output, sample_rate, cfg, settings["baseband_offset_hz"])
                print("QUALITY:", json.dumps(quality, ensure_ascii=False))
                if (not quality_gate) or bool(quality["quality_pass"]):
                    break
                if attempt + 1 < max_retries:
                    time.sleep(0.5)
            captures.append(
                {
                    "file": str(output),
                    "modulation": modulation,
                    "capture_index": capture_idx,
                    "payload_seed": payload_seed,
                    "random_payload_bits": True,
                    "quality": quality,
                }
            )
    metadata = {
        "experiment": "Oshea2018",
        "session_id": session_id,
        "hardware_change_from_paper": "USRP_B210_TX_RX replaced by HackRF_One_TX and RTL_SDR_RX",
        "modulation_change_from_paper": "24-class/11-class replaced by BASK/BFSK/BPSK",
        "center_freq": float(cfg["sdr"]["center_freq"]),
        "sample_rate": sample_rate,
        "symbol_rate": float(cfg["sdr"]["symbol_rate"]),
        "distance_m": float(cfg["sdr"]["distance_m"]),
        "baseband_offset_hz": float(settings["baseband_offset_hz"]),
        "tx_vga_gain": float(settings["tx_vga_gain"]),
        "tx_amp_gain": float(settings["tx_amp_gain"]),
        "rx_gain": float(settings["rx_gain"]),
        "tx_backend": tx_backend_value,
        "agc": bool(cfg["sdr"]["agc"]),
        "noise_only_file": str(noise_output),
        "random_payload_bits": True,
        "payload_seed_policy": "same seed for BASK/BFSK/BPSK at the same session/capture index",
        "captures": captures,
    }
    if dry_run:
        print(f"Dry run complete. Metadata would be written to {session_dir / 'metadata.json'}")
        return session_dir
    write_json(session_dir / "metadata.json", metadata)
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--output-root", default="../data/raw_ota_clean")
    parser.add_argument("--radioconda-root", default=None)
    parser.add_argument("--captures-per-class", type=int, default=10)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--rx-gain", type=float, default=None)
    parser.add_argument("--baseband-offset-hz", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-quality-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--process-timeout-margin-seconds", type=float, default=60.0)
    parser.add_argument("--tx-backend", choices=["hackrf_transfer", "gnuradio", "soapy"], default=None)
    args = parser.parse_args()
    run_oshea2018_capture_session(
        args.session_id,
        args.config,
        args.output_root,
        args.radioconda_root,
        args.captures_per_class,
        args.tx_vga_gain,
        args.tx_amp_gain,
        args.rx_gain,
        args.baseband_offset_hz,
        args.max_retries,
        not args.no_quality_gate,
        args.dry_run,
        args.process_timeout_margin_seconds,
        args.tx_backend,
    )


if __name__ == "__main__":
    main()
