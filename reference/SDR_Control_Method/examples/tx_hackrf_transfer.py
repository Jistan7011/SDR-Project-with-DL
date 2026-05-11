from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def random_bits(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=count, dtype=np.uint8)


def generate_iq(
    modulation: str,
    seconds: float,
    sample_rate: float,
    symbol_rate: float,
    baseband_offset_hz: float,
    seed: int,
    bfsk_freq_dev_hz: float,
) -> np.ndarray:
    sps = max(1, int(round(sample_rate / symbol_rate)))
    bit_count = max(256, int(np.ceil(seconds * symbol_rate)))
    bits = random_bits(seed, bit_count)
    mod = modulation.upper()

    if mod == "BASK":
        symbols = np.where(bits > 0, 1.0, 0.15).astype(np.float32)
        baseband = np.repeat(symbols, sps).astype(np.complex64)
    elif mod == "BPSK":
        symbols = np.where(bits > 0, -1.0, 1.0).astype(np.float32)
        baseband = np.repeat(symbols, sps).astype(np.complex64)
    elif mod == "BFSK":
        freqs = np.repeat(np.where(bits > 0, bfsk_freq_dev_hz, -bfsk_freq_dev_hz), sps)
        phase = np.cumsum(2.0 * np.pi * freqs / sample_rate)
        baseband = np.exp(1j * phase).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    total = int(round(sample_rate * seconds))
    if len(baseband) < total:
        repeats = int(np.ceil(total / len(baseband)))
        baseband = np.tile(baseband, repeats)
    baseband = baseband[:total]

    n = np.arange(len(baseband), dtype=np.float64)
    shifted = baseband * np.exp(1j * 2.0 * np.pi * baseband_offset_hz * n / sample_rate)
    shifted = shifted / (np.max(np.abs(shifted)) + 1e-9) * 0.5
    return shifted.astype(np.complex64)


def complex64_to_cs8(iq: np.ndarray) -> np.ndarray:
    clipped = np.clip(iq, -0.95 - 0.95j, 0.95 + 0.95j)
    out = np.empty(len(clipped) * 2, dtype=np.int8)
    out[0::2] = np.clip(np.round(clipped.real * 127.0), -128, 127).astype(np.int8)
    out[1::2] = np.clip(np.round(clipped.imag * 127.0), -128, 127).astype(np.int8)
    return out


def transmit(
    modulation: str,
    seconds: float,
    sample_rate: float,
    center_freq: float,
    symbol_rate: float,
    baseband_offset_hz: float,
    tx_vga_gain: float,
    tx_amp_gain: float,
    seed: int,
    bfsk_freq_dev_hz: float,
) -> None:
    iq = generate_iq(
        modulation=modulation,
        seconds=seconds,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        baseband_offset_hz=baseband_offset_hz,
        seed=seed,
        bfsk_freq_dev_hz=bfsk_freq_dev_hz,
    )
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="sdr_tx_", suffix=".cs8", delete=False) as handle:
            tmp_path = Path(handle.name)
            complex64_to_cs8(iq).tofile(handle)
        command = [
            "hackrf_transfer",
            "-t",
            str(tmp_path),
            "-f",
            str(int(round(center_freq))),
            "-s",
            str(int(round(sample_rate))),
            "-x",
            str(int(round(tx_vga_gain))),
            "-a",
            "1" if tx_amp_gain > 0 else "0",
            "-n",
            str(len(iq)),
            "-R",
        ]
        print("TX:", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(f"hackrf_transfer failed: {result.returncode}")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Transmit BASK/BFSK/BPSK with HackRF using hackrf_transfer.")
    parser.add_argument("--modulation", choices=["BASK", "BFSK", "BPSK"], required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=2_400_000.0)
    parser.add_argument("--center-freq", type=float, default=433_920_000.0)
    parser.add_argument("--symbol-rate", type=float, default=5_000.0)
    parser.add_argument("--baseband-offset-hz", type=float, default=500_000.0)
    parser.add_argument("--tx-vga-gain", type=float, default=30.0)
    parser.add_argument("--tx-amp-gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bfsk-freq-dev-hz", type=float, default=50_000.0)
    args = parser.parse_args()
    transmit(
        modulation=args.modulation,
        seconds=args.seconds,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        symbol_rate=args.symbol_rate,
        baseband_offset_hz=args.baseband_offset_hz,
        tx_vga_gain=args.tx_vga_gain,
        tx_amp_gain=args.tx_amp_gain,
        seed=args.seed,
        bfsk_freq_dev_hz=args.bfsk_freq_dev_hz,
    )


if __name__ == "__main__":
    main()
