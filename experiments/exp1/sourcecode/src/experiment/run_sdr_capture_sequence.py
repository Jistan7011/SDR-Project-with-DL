from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SEQUENCE = [
    ("BASK", "A", "bask_a.bin"),
    ("BFSK", "F", "bfsk_f.bin"),
    ("BPSK", "P", "bpsk_p.bin"),
]


def radioconda_cmd(radioconda_root: Path, module: str, args: list[str]) -> list[str]:
    activate = radioconda_root / "Scripts" / "activate.bat"
    return ["cmd", "/c", "call", str(activate), str(radioconda_root), "&&", "python", "-m", module, *args]


def quote_arg(value: str) -> str:
    if any(ch in value for ch in (' ', '&', '(', ')', '^', '"')):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def run_one(
    radioconda_root: Path,
    modulation: str,
    payload: str,
    output: Path,
    config: Path,
    capture_seconds: float,
    tx_seconds: float,
    tx_delay: float,
    dry_run: bool,
    tx_amp_gain: float | None,
    tx_vga_gain: float | None,
) -> None:
    capture_cmd = radioconda_cmd(
        radioconda_root,
        "src.sdr.capture_iq",
        ["--output", str(output), "--config", str(config), "--seconds", str(capture_seconds)],
    )
    tx_cmd = radioconda_cmd(
        radioconda_root,
        "src.sdr.hackrf_tx",
        ["--modulation", modulation, "--payload", payload, "--config", str(config), "--seconds", str(tx_seconds)],
    )
    if tx_amp_gain is not None:
        tx_cmd.extend(["--tx-amp-gain", str(tx_amp_gain)])
    if tx_vga_gain is not None:
        tx_cmd.extend(["--tx-vga-gain", str(tx_vga_gain)])

    print(f"\n=== {modulation} / {payload} ===")
    print("RX:", " ".join(quote_arg(part) for part in capture_cmd))
    print("TX:", " ".join(quote_arg(part) for part in tx_cmd))
    if dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    rx_proc = subprocess.Popen(capture_cmd)
    try:
        time.sleep(tx_delay)
        tx_result = subprocess.run(tx_cmd)
        if tx_result.returncode != 0:
            raise SystemExit(f"TX failed for {modulation}/{payload} with exit code {tx_result.returncode}")
        rx_code = rx_proc.wait(timeout=capture_seconds + 10)
        if rx_code != 0:
            raise SystemExit(f"RX failed for {modulation}/{payload} with exit code {rx_code}")
    except BaseException:
        if rx_proc.poll() is None:
            rx_proc.terminate()
            try:
                rx_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rx_proc.kill()
        raise


def import_capture(input_path: Path, modulation: str, payload: str, config: Path, import_seconds: float, import_skip: float, max_windows: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.dataset.import_real_iq",
        "--input",
        str(input_path),
        "--modulation",
        modulation,
        "--payload",
        payload,
        "--config",
        str(config),
        "--skip-seconds",
        str(import_skip),
        "--duration-seconds",
        str(import_seconds),
        "--max-windows",
        str(max_windows),
    ]
    print("IMPORT:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Import failed for {input_path} with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate RTL-SDR capture + HackRF TX for BASK/BFSK/BPSK.")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--config", default=r"..\config\config.exp01.yaml")
    parser.add_argument("--output-dir", default=r"data\real\raw_iq")
    parser.add_argument("--capture-seconds", type=float, default=8.0)
    parser.add_argument("--tx-seconds", type=float, default=3.0)
    parser.add_argument("--tx-delay", type=float, default=1.0)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    parser.add_argument("--import-after", action="store_true", help="Convert captured .bin files to labeled .npz windows.")
    parser.add_argument("--import-skip-seconds", type=float, default=1.0)
    parser.add_argument("--import-duration-seconds", type=float, default=4.0)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    radioconda_root = Path(args.radioconda_root)
    config = Path(args.config)
    output_dir = Path(args.output_dir)
    if not args.dry_run and not (radioconda_root / "Scripts" / "activate.bat").exists():
        raise SystemExit(f"Radioconda activate.bat not found under {radioconda_root}")

    for modulation, payload, filename in DEFAULT_SEQUENCE:
        output = output_dir / filename
        run_one(
            radioconda_root=radioconda_root,
            modulation=modulation,
            payload=payload,
            output=output,
            config=config,
            capture_seconds=args.capture_seconds,
            tx_seconds=args.tx_seconds,
            tx_delay=args.tx_delay,
            dry_run=args.dry_run,
            tx_amp_gain=args.tx_amp_gain,
            tx_vga_gain=args.tx_vga_gain,
        )
        if args.import_after and not args.dry_run:
            import_capture(output, modulation, payload, config, args.import_duration_seconds, args.import_skip_seconds, args.max_windows)

    print("\nSequence complete.")


if __name__ == "__main__":
    main()
