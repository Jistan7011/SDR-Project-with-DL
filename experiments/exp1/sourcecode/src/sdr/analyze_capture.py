from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import ensure_dir, load_config, write_json


def analyze_capture(input_path: str, config_path: str = "config.yaml", block_size: int = 24000, max_fft_samples: int = 262144) -> dict[str, object]:
    cfg = load_config(config_path)
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    path = Path(input_path)
    iq = np.fromfile(path, dtype=np.complex64)
    if len(iq) == 0:
        raise SystemExit(f"No samples in {path}")

    n_blocks = len(iq) // block_size
    trimmed = iq[: n_blocks * block_size] if n_blocks else iq
    blocks = trimmed.reshape(n_blocks, block_size) if n_blocks else trimmed.reshape(1, -1)
    power = np.mean(np.abs(blocks) ** 2, axis=1)
    rms = np.sqrt(power)
    times = (np.arange(len(rms)) * block_size) / sample_rate

    fft_len = min(max_fft_samples, len(iq))
    fft_iq = iq[:fft_len] - np.mean(iq[:fft_len])
    window = np.hanning(len(fft_iq))
    spec = np.fft.fftshift(np.fft.fft(fft_iq * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fft_iq), d=1.0 / sample_rate))
    mag_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
    peak_index = int(np.argmax(mag_db))
    peak_freq_hz = float(freqs[peak_index])
    peak_db = float(mag_db[peak_index])

    out_dir = ensure_dir("results/real_capture_analysis")
    stem = path.stem
    plt.figure(figsize=(8, 3))
    plt.plot(times, rms)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS")
    plt.title(f"RMS over time: {stem}")
    plt.grid(True)
    plt.tight_layout()
    rms_png = out_dir / f"{stem}_rms.png"
    plt.savefig(rms_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(freqs / 1000.0, mag_db)
    plt.xlabel("Frequency offset (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectrum: {stem}")
    plt.grid(True)
    plt.tight_layout()
    spectrum_png = out_dir / f"{stem}_spectrum.png"
    plt.savefig(spectrum_png, dpi=150)
    plt.close()

    summary = {
        "input": str(path),
        "samples": int(len(iq)),
        "duration_seconds": float(len(iq) / sample_rate),
        "rms_min": float(np.min(rms)),
        "rms_max": float(np.max(rms)),
        "rms_mean": float(np.mean(rms)),
        "rms_p95": float(np.percentile(rms, 95)),
        "peak_freq_hz": peak_freq_hz,
        "peak_freq_khz": peak_freq_hz / 1000.0,
        "peak_db": peak_db,
        "rms_plot": str(rms_png),
        "spectrum_plot": str(spectrum_png),
    }
    write_json(out_dir / f"{stem}_summary.json", summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--block-size", type=int, default=24000)
    args = parser.parse_args()
    analyze_capture(args.input, args.config, args.block_size)


if __name__ == "__main__":
    main()
