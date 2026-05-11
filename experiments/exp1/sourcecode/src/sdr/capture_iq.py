from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common import ensure_dir, load_config
from src.sdr.soapy_common import require_soapy


def capture(output: str, config_path: str = "config.yaml", seconds: float = 3.0) -> Path:
    SoapySDR, SOAPY_SDR_CF32, SOAPY_SDR_RX, _ = require_soapy()
    cfg = load_config(config_path)
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    total = int(sample_rate * seconds)
    out = Path(output)
    ensure_dir(out.parent)
    try:
        sdr = SoapySDR.Device("driver=rtlsdr")
    except Exception as exc:
        raise SystemExit(f"Could not open RTL-SDR via SoapySDR: {exc}") from exc
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, float(cfg["sdr"]["center_freq"]))
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, "TUNER", float(cfg["sdr"]["rx_gain"]))
    except Exception:
        sdr.setGain(SOAPY_SDR_RX, 0, float(cfg["sdr"]["rx_gain"]))
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
    sdr.activateStream(stream)
    data = np.empty(total, dtype=np.complex64)
    offset = 0
    while offset < total:
        chunk = np.empty(min(4096, total - offset), dtype=np.complex64)
        sr = sdr.readStream(stream, [chunk], len(chunk))
        if sr.ret > 0:
            data[offset : offset + sr.ret] = chunk[: sr.ret]
            offset += sr.ret
        elif sr.ret < 0:
            raise SystemExit(f"SoapySDR readStream failed with code {sr.ret}")
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)
    data.tofile(out)
    print(f"Captured {len(data)} complex64 samples to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    capture(args.output, args.config, args.seconds)


if __name__ == "__main__":
    main()
