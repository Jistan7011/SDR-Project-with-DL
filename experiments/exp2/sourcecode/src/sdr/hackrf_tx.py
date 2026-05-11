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
    baseband_offset_hz: float | None = None,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> None:
    SoapySDR, SOAPY_SDR_CF32, _, SOAPY_SDR_TX = require_soapy()
    cfg = load_config(config_path)
    sample_rate = float(cfg["sdr"]["tx_sample_rate"] if sample_rate is None else sample_rate)
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    sps = max(1, int(round(sample_rate / symbol_rate)))
    iq = modulate_bits(modulation, make_frame(payload), sps, sample_rate)
    if baseband_offset_hz is not None:
        # Existing baseband modulators are centered near 100 kHz. Shift the
        # whole waveform so BASK/BPSK carrier and BFSK center move together.
        default_center_hz = 100_000.0
        shift_hz = float(baseband_offset_hz) - default_center_hz
        t = np.arange(len(iq), dtype=np.float64) / sample_rate
        iq = iq * np.exp(1j * 2.0 * np.pi * shift_hz * t)
    iq = (iq / (np.max(np.abs(iq)) + 1e-9) * 0.5).astype(np.complex64)

    try:
        sdr = SoapySDR.Device("driver=hackrf")
    except Exception as exc:
        raise SystemExit(f"Could not open HackRF via SoapySDR: {exc}") from exc
    sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, 0, float(cfg["sdr"]["center_freq"] if center_freq is None else center_freq))
    amp_gain = float(cfg["sdr"].get("tx_amp_gain", 0.0) if tx_amp_gain is None else tx_amp_gain)
    vga_gain = float(cfg["sdr"].get("tx_vga_gain", cfg["sdr"].get("tx_gain", 0.0)) if tx_vga_gain is None else tx_vga_gain)
    try:
        sdr.setGain(SOAPY_SDR_TX, 0, "AMP", amp_gain)
        sdr.setGain(SOAPY_SDR_TX, 0, "VGA", vga_gain)
    except Exception:
        sdr.setGain(SOAPY_SDR_TX, 0, vga_gain)
    offset_msg = "" if baseband_offset_hz is None else f" baseband_offset_hz={float(baseband_offset_hz):.1f}"
    print(f"TX configured: modulation={modulation} payload={payload} amp_gain={amp_gain} vga_gain={vga_gain}{offset_msg}")
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
    parser.add_argument("--baseband-offset-hz", type=float, default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--center-freq", type=float, default=None)
    args = parser.parse_args()
    transmit(
        args.modulation,
        args.payload,
        args.config,
        args.seconds,
        args.tx_amp_gain,
        args.tx_vga_gain,
        args.baseband_offset_hz,
        args.sample_rate,
        args.center_freq,
    )


if __name__ == "__main__":
    main()
