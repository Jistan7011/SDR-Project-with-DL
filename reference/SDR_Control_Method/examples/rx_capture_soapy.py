from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def capture(
    output: Path,
    seconds: float,
    sample_rate: float,
    center_freq: float,
    rx_gain: float,
) -> Path:
    try:
        import SoapySDR  # type: ignore
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore
    except Exception as exc:
        raise SystemExit(f"SoapySDR Python binding is unavailable: {exc}") from exc

    total = int(round(sample_rate * seconds))
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        sdr = SoapySDR.Device("driver=rtlsdr")
    except Exception as exc:
        raise SystemExit(f"Could not open RTL-SDR via SoapySDR: {exc}") from exc

    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, "TUNER", rx_gain)
    except Exception:
        sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
    sdr.activateStream(stream)
    data = np.empty(total, dtype=np.complex64)
    offset = 0
    try:
        while offset < total:
            chunk = np.empty(min(4096, total - offset), dtype=np.complex64)
            result = sdr.readStream(stream, [chunk], len(chunk))
            if result.ret > 0:
                data[offset : offset + result.ret] = chunk[: result.ret]
                offset += result.ret
            elif result.ret < 0:
                raise SystemExit(f"SoapySDR readStream failed: {result.ret}")
    finally:
        try:
            sdr.deactivateStream(stream)
        finally:
            sdr.closeStream(stream)

    data.tofile(output)
    print(f"Captured {len(data)} complex64 samples to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture complex64 IQ with RTL-SDR via SoapySDR.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=float, default=2_400_000.0)
    parser.add_argument("--center-freq", type=float, default=433_920_000.0)
    parser.add_argument("--rx-gain", type=float, default=30.0)
    args = parser.parse_args()
    capture(args.output, args.seconds, args.sample_rate, args.center_freq, args.rx_gain)


if __name__ == "__main__":
    main()
