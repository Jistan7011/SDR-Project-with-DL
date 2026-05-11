from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json


DEFAULT_PAYLOAD_POOL = ["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]


def run_session(
    session_id: str,
    output_root: str = "../data/raw_iq",
    config_path: str = "../config/config.exp02.yaml",
    radioconda_root: str = r"C:\Users\qus70\radioconda",
    payloads: list[str] | None = None,
    baseband_offset_hz: float = 250_000.0,
    tx_vga_gain: float | None = None,
    tx_amp_gain: float | None = None,
    rx_gain: float | None = None,
    tx_seconds: float = 5.0,
    capture_seconds: float = 10.0,
    noise_seconds: float = 3.0,
    tx_start_delay_seconds: float = 1.0,
    sample_rate: float | None = None,
    dry_run: bool = False,
) -> Path:
    cfg = load_config(config_path)
    payload_pool = payloads or list(cfg.get("experiment2", {}).get("payload_pool", DEFAULT_PAYLOAD_POOL))
    session_dir = ensure_dir(Path(output_root) / session_id)
    tx_vga = float(cfg["sdr"].get("tx_vga_gain", 30) if tx_vga_gain is None else tx_vga_gain)
    tx_amp = float(cfg["sdr"].get("tx_amp_gain", 0) if tx_amp_gain is None else tx_amp_gain)
    rx_gain_value = float(cfg["sdr"].get("rx_gain", 30) if rx_gain is None else rx_gain)
    rx_sample_rate = float(cfg["sdr"].get("rx_sample_rate", 2.4e6) if sample_rate is None else sample_rate)

    captures: list[dict[str, object]] = []
    noise_path = session_dir / "noise_only.bin"
    noise_cmd = radioconda_python_cmd(
        radioconda_root,
        ["-m", "src.sdr.capture_iq", "--config", config_path, "--output", str(noise_path), "--seconds", str(noise_seconds), "--sample-rate", str(rx_sample_rate), "--rx-gain", str(rx_gain_value)],
    )
    run_or_print(noise_cmd, dry_run)

    for payload in payload_pool:
        for modulation in CLASS_NAMES:
            filename = f"{modulation.lower()}_{payload}_offset{int(baseband_offset_hz)}_txvga{int(tx_vga)}.bin"
            output = session_dir / filename
            rx_cmd = radioconda_python_cmd(
                radioconda_root,
                ["-m", "src.sdr.capture_iq", "--config", config_path, "--output", str(output), "--seconds", str(capture_seconds), "--sample-rate", str(rx_sample_rate), "--rx-gain", str(rx_gain_value)],
            )
            tx_cmd = radioconda_python_cmd(
                radioconda_root,
                [
                    "-m",
                    "src.sdr.hackrf_tx",
                    "--config",
                    config_path,
                    "--modulation",
                    modulation,
                    "--payload",
                    payload,
                    "--seconds",
                    str(tx_seconds),
                    "--tx-amp-gain",
                    str(tx_amp),
                    "--tx-vga-gain",
                    str(tx_vga),
                    "--baseband-offset-hz",
                    str(baseband_offset_hz),
                ],
            )
            print(f"\n{session_id}: {modulation}/{payload}")
            print("RX:", " ".join(rx_cmd))
            print("TX:", " ".join(tx_cmd))
            if dry_run:
                continue
            rx_proc = subprocess.Popen(rx_cmd)
            time.sleep(tx_start_delay_seconds)
            tx_proc = subprocess.Popen(tx_cmd)
            tx_code = tx_proc.wait()
            rx_code = rx_proc.wait()
            if tx_code != 0:
                raise SystemExit(f"TX failed for {modulation}/{payload}: {tx_code}")
            if rx_code != 0:
                raise SystemExit(f"RX failed for {modulation}/{payload}: {rx_code}")
            captures.append({"file": str(output), "modulation": modulation, "payload": payload})

    metadata = {
        "session_id": session_id,
        "definition": "Independent RF condition group; windows from this session must not cross train/val/test splits.",
        "noise_only_file": str(noise_path),
        "payload_pool": payload_pool,
        "baseband_offset_hz": baseband_offset_hz,
        "center_freq": float(cfg["sdr"]["center_freq"]),
        "tx_sample_rate": float(cfg["sdr"]["tx_sample_rate"]),
        "rx_sample_rate": rx_sample_rate,
        "symbol_rate": float(cfg["sdr"]["symbol_rate"]),
        "tx_vga_gain": tx_vga,
        "tx_amp_gain": tx_amp,
        "rx_gain": rx_gain_value,
        "tx_seconds": tx_seconds,
        "capture_seconds": capture_seconds,
        "noise_seconds": noise_seconds,
        "captures": captures,
    }
    if dry_run:
        print(f"Dry run complete. Metadata would be written to {session_dir / 'metadata.json'}")
        return session_dir
    write_json(session_dir / "metadata.json", metadata)
    print(f"Session metadata written to {session_dir / 'metadata.json'}")
    return session_dir


def radioconda_python_cmd(radioconda_root: str, python_args: list[str]) -> list[str]:
    activate = str(Path(radioconda_root) / "Scripts" / "activate.bat")
    return ["cmd", "/c", "call", activate, radioconda_root, "&&", "python", *python_args]


def run_or_print(command: list[str], dry_run: bool) -> None:
    print("NOISE:", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", default="../data/raw_iq")
    parser.add_argument("--config", default="../config/config.exp02.yaml")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--payloads", nargs="*", default=None)
    parser.add_argument("--baseband-offset-hz", type=float, default=250_000.0)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--rx-gain", type=float, default=None)
    parser.add_argument("--tx-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--noise-seconds", type=float, default=3.0)
    parser.add_argument("--tx-start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_session(
        session_id=args.session_id,
        output_root=args.output_root,
        config_path=args.config,
        radioconda_root=args.radioconda_root,
        payloads=args.payloads,
        baseband_offset_hz=args.baseband_offset_hz,
        tx_vga_gain=args.tx_vga_gain,
        tx_amp_gain=args.tx_amp_gain,
        rx_gain=args.rx_gain,
        tx_seconds=args.tx_seconds,
        capture_seconds=args.capture_seconds,
        noise_seconds=args.noise_seconds,
        tx_start_delay_seconds=args.tx_start_delay_seconds,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
