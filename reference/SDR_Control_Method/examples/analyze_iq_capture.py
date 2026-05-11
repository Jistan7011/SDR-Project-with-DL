from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def clipping_rate(x: np.ndarray, threshold: float = 0.98) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean((np.abs(x.real) >= threshold) | (np.abs(x.imag) >= threshold)))


def spectral_peak(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    if len(x) == 0:
        return {"peak_freq_hz": 0.0, "peak_to_median": 0.0}
    n = min(len(x), 1_048_576)
    y = x[:n] - np.mean(x[:n])
    spec = np.abs(np.fft.fftshift(np.fft.fft(y)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    idx = int(np.argmax(spec))
    return {
        "peak_freq_hz": float(freqs[idx]),
        "peak_to_median": float(spec[idx] / (np.median(spec) + 1e-12)),
    }


def analyze(
    noise_path: Path,
    capture_path: Path,
    sample_rate: float,
    active_start_seconds: float,
    active_duration_seconds: float,
    output: Path | None,
) -> dict[str, object]:
    noise = np.fromfile(noise_path, dtype=np.complex64)
    capture = np.fromfile(capture_path, dtype=np.complex64)
    start = min(len(capture), int(round(active_start_seconds * sample_rate)))
    end = min(len(capture), start + int(round(active_duration_seconds * sample_rate)))
    active = capture[start:end]

    noise_rms = rms(noise)
    capture_rms = rms(active)
    ratio = capture_rms / max(noise_rms, 1e-12)
    snr_proxy_db = 20.0 * np.log10(max(ratio, 1e-12))
    result: dict[str, object] = {
        "noise_file": str(noise_path),
        "capture_file": str(capture_path),
        "noise_samples": int(len(noise)),
        "capture_samples": int(len(capture)),
        "active_samples": int(len(active)),
        "noise_rms": noise_rms,
        "capture_active_rms": capture_rms,
        "tx_to_noise_rms_ratio": float(ratio),
        "snr_proxy_db": float(snr_proxy_db),
        "clipping_rate": clipping_rate(active),
        "spectral_peak": spectral_peak(active, sample_rate),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze noise/capture complex64 IQ files.")
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=2_400_000.0)
    parser.add_argument("--active-start-seconds", type=float, default=1.1)
    parser.add_argument("--active-duration-seconds", type=float, default=3.8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    analyze(
        noise_path=args.noise,
        capture_path=args.capture,
        sample_rate=args.sample_rate,
        active_start_seconds=args.active_start_seconds,
        active_duration_seconds=args.active_duration_seconds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
