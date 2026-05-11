from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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


def run_capture(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_path = output_dir / "noise_only.bin"
    capture_path = output_dir / f"{args.modulation.lower()}_capture.bin"
    metadata_path = output_dir / "metadata.json"

    common = [
        "--sample-rate",
        str(args.sample_rate),
        "--center-freq",
        str(args.center_freq),
    ]
    noise_cmd = [
        sys.executable,
        str(Path(__file__).with_name("rx_capture_soapy.py")),
        "--output",
        str(noise_path),
        "--seconds",
        str(args.noise_seconds),
        "--rx-gain",
        str(args.rx_gain),
        *common,
    ]
    rx_cmd = [
        sys.executable,
        str(Path(__file__).with_name("rx_capture_soapy.py")),
        "--output",
        str(capture_path),
        "--seconds",
        str(args.seconds),
        "--rx-gain",
        str(args.rx_gain),
        *common,
    ]
    tx_cmd = [
        sys.executable,
        str(Path(__file__).with_name("tx_hackrf_transfer.py")),
        "--modulation",
        args.modulation,
        "--seconds",
        str(args.tx_seconds),
        "--symbol-rate",
        str(args.symbol_rate),
        "--baseband-offset-hz",
        str(args.baseband_offset_hz),
        "--tx-vga-gain",
        str(args.tx_vga_gain),
        "--tx-amp-gain",
        str(args.tx_amp_gain),
        "--seed",
        str(args.seed),
        *common,
    ]

    print("NOISE:", " ".join(noise_cmd))
    noise_code = subprocess.run(noise_cmd, check=False).returncode
    if noise_code != 0:
        raise SystemExit(f"noise capture failed: {noise_code}")

    print("RX:", " ".join(rx_cmd))
    print("TX:", " ".join(tx_cmd))
    rx_proc = subprocess.Popen(rx_cmd)
    time.sleep(args.rx_lead_seconds)
    tx_proc = subprocess.Popen(tx_cmd)
    try:
        tx_code = wait_process(tx_proc, args.tx_seconds + args.timeout_margin_seconds, "TX")
        rx_code = wait_process(rx_proc, args.seconds + args.timeout_margin_seconds, "RX")
    except BaseException:
        for proc in (tx_proc, rx_proc):
            if proc.poll() is None:
                terminate_process_tree(proc)
        raise
    if tx_code != 0:
        raise SystemExit(f"TX failed: {tx_code}")
    if rx_code != 0:
        raise SystemExit(f"RX failed: {rx_code}")

    metadata = {
        "modulation": args.modulation,
        "noise_file": str(noise_path),
        "capture_file": str(capture_path),
        "center_freq": args.center_freq,
        "sample_rate": args.sample_rate,
        "symbol_rate": args.symbol_rate,
        "baseband_offset_hz": args.baseband_offset_hz,
        "rx_gain": args.rx_gain,
        "tx_vga_gain": args.tx_vga_gain,
        "tx_amp_gain": args.tx_amp_gain,
        "seed": args.seed,
        "seconds": args.seconds,
        "tx_seconds": args.tx_seconds,
        "rx_lead_seconds": args.rx_lead_seconds,
        "tx_backend": "hackrf_transfer",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata written: {metadata_path}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RTL-SDR RX + HackRF TX OTA capture.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modulation", choices=["BASK", "BFSK", "BPSK"], required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--tx-seconds", type=float, default=4.5)
    parser.add_argument("--noise-seconds", type=float, default=2.0)
    parser.add_argument("--rx-lead-seconds", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=float, default=2_400_000.0)
    parser.add_argument("--center-freq", type=float, default=433_920_000.0)
    parser.add_argument("--symbol-rate", type=float, default=5_000.0)
    parser.add_argument("--baseband-offset-hz", type=float, default=500_000.0)
    parser.add_argument("--rx-gain", type=float, default=30.0)
    parser.add_argument("--tx-vga-gain", type=float, default=30.0)
    parser.add_argument("--tx-amp-gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-margin-seconds", type=float, default=30.0)
    args = parser.parse_args()
    run_capture(args)


if __name__ == "__main__":
    main()
