from __future__ import annotations

import argparse
import time

import numpy as np

from src.common import load_config
from src.sdr.soapy_common import require_soapy
from src.signal.frame import make_frame
from src.signal.modulate import modulate_bits


def transmit(
    modulation: str,
    payload: str,
    config_path: str = "config.yaml",
    seconds: float = 3.0,
    tx_amp_gain: float | None = None,
    tx_vga_gain: float | None = None,
) -> None:
    SoapySDR, SOAPY_SDR_CF32, _, SOAPY_SDR_TX = require_soapy()
    cfg = load_config(config_path)
    sample_rate = float(cfg["sdr"]["tx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    sps = max(1, int(round(sample_rate / symbol_rate)))
    iq = modulate_bits(modulation, make_frame(payload), sps, sample_rate)
    iq = (iq / (np.max(np.abs(iq)) + 1e-9) * 0.5).astype(np.complex64)

    try:
        sdr = SoapySDR.Device("driver=hackrf")
    except Exception as exc:
        raise SystemExit(f"Could not open HackRF via SoapySDR: {exc}") from exc
    sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, 0, float(cfg["sdr"]["center_freq"]))
    amp_gain = float(cfg["sdr"].get("tx_amp_gain", 0.0) if tx_amp_gain is None else tx_amp_gain)
    vga_gain = float(cfg["sdr"].get("tx_vga_gain", cfg["sdr"].get("tx_gain", 0.0)) if tx_vga_gain is None else tx_vga_gain)
    try:
        sdr.setGain(SOAPY_SDR_TX, 0, "AMP", amp_gain)
        sdr.setGain(SOAPY_SDR_TX, 0, "VGA", vga_gain)
    except Exception:
        sdr.setGain(SOAPY_SDR_TX, 0, vga_gain)
    print(f"TX configured: modulation={modulation} payload={payload} amp_gain={amp_gain} vga_gain={vga_gain}")
    stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
    sdr.activateStream(stream)
    deadline = time.time() + seconds
    index = 0
    while time.time() < deadline:
        chunk = iq[index : index + 4096]
        if len(chunk) == 0:
            index = 0
            continue
        sdr.writeStream(stream, [chunk], len(chunk))
        index += len(chunk)
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulation", choices=["BASK", "BFSK", "BPSK"], required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    args = parser.parse_args()
    transmit(args.modulation, args.payload, args.config, args.seconds, args.tx_amp_gain, args.tx_vga_gain)


if __name__ == "__main__":
    main()
