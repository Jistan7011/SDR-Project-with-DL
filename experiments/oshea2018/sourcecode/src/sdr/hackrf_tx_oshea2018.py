from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from src.common import load_config
from src.sdr.soapy_common import require_soapy
from src.signal.oshea2018_waveform import generate_clean_modulation, random_bits


def transmit_oshea2018(
    modulation: str,
    config_path: str = "../config/config.oshea2018.yaml",
    seconds: float = 5.0,
    seed: int = 42,
    tx_vga_gain: float | None = None,
    tx_amp_gain: float | None = None,
    baseband_offset_hz: float | None = None,
    backend: str = "hackrf_transfer",
) -> None:
    cfg = load_config(config_path)
    iq, sample_rate, offset = build_tx_iq(modulation, cfg, seconds, seed, baseband_offset_hz)
    amp_gain = float(cfg["sdr"].get("tx_amp_gain", 0.0) if tx_amp_gain is None else tx_amp_gain)
    vga_gain = float(cfg["sdr"].get("tx_vga_gain", 0.0) if tx_vga_gain is None else tx_vga_gain)
    center_freq = float(cfg["sdr"]["center_freq"])
    print(
        f"TX Oshea2018 modulation={modulation} backend={backend} "
        f"sample_rate={sample_rate} center_freq={center_freq} offset_hz={offset} amp={amp_gain} vga={vga_gain}"
    )
    if backend == "hackrf_transfer":
        transmit_with_hackrf_transfer(iq, sample_rate, center_freq, seconds, amp_gain, vga_gain)
        return
    if backend == "gnuradio":
        transmit_with_gnuradio(iq, sample_rate, center_freq, seconds, amp_gain, vga_gain)
        return
    if backend == "soapy":
        transmit_with_soapy(iq, sample_rate, center_freq, seconds, amp_gain, vga_gain)
        return
    raise ValueError(f"Unsupported TX backend: {backend}")


def build_tx_iq(
    modulation: str,
    cfg: dict,
    seconds: float,
    seed: int,
    baseband_offset_hz: float | None,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    sample_rate = float(cfg["sdr"]["tx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    bit_count = max(256, int(seconds * symbol_rate))
    bits = random_bits(rng, bit_count)
    iq = generate_clean_modulation(
        modulation,
        bits,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        rolloff=0.35,
        bfsk_freq_dev_hz=float(cfg["modulation"]["bfsk_freq_dev_hz"]),
    )
    offset = float(cfg["sdr"]["baseband_offset_hz"] if baseband_offset_hz is None else baseband_offset_hz)
    n = np.arange(len(iq), dtype=np.float64)
    iq = iq * np.exp(1j * 2.0 * np.pi * offset * n / sample_rate)
    iq = (iq / (np.max(np.abs(iq)) + 1e-9) * 0.5).astype(np.complex64)
    return iq, sample_rate, offset


def transmit_with_soapy(
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    seconds: float,
    amp_gain: float,
    vga_gain: float,
) -> None:
    SoapySDR, SOAPY_SDR_CF32, _, SOAPY_SDR_TX = require_soapy()
    try:
        sdr = SoapySDR.Device("driver=hackrf")
    except Exception as exc:
        raise SystemExit(f"Could not open HackRF via SoapySDR: {exc}") from exc
    sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
    try:
        sdr.setGain(SOAPY_SDR_TX, 0, "AMP", amp_gain)
        sdr.setGain(SOAPY_SDR_TX, 0, "VGA", vga_gain)
    except Exception:
        sdr.setGain(SOAPY_SDR_TX, 0, vga_gain)
    stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
    try:
        sdr.activateStream(stream)
        deadline = time.time() + seconds
        index = 0
        while time.time() < deadline:
            chunk = iq[index : index + 4096]
            if len(chunk) == 0:
                index = 0
                continue
            result = sdr.writeStream(stream, [chunk], len(chunk), timeoutUs=1_000_000)
            written = getattr(result, "ret", result if isinstance(result, int) else len(chunk))
            if int(written) < 0:
                raise SystemExit(f"HackRF writeStream failed: {written}")
            if int(written) == 0:
                time.sleep(0.001)
                continue
            index += int(written)
    finally:
        try:
            sdr.deactivateStream(stream)
        finally:
            sdr.closeStream(stream)


def complex64_to_cs8(iq: np.ndarray) -> np.ndarray:
    clipped = np.clip(iq, -0.95 - 0.95j, 0.95 + 0.95j)
    interleaved = np.empty(len(clipped) * 2, dtype=np.int8)
    interleaved[0::2] = np.clip(np.round(clipped.real * 127.0), -128, 127).astype(np.int8)
    interleaved[1::2] = np.clip(np.round(clipped.imag * 127.0), -128, 127).astype(np.int8)
    return interleaved


def transmit_with_hackrf_transfer(
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    seconds: float,
    amp_gain: float,
    vga_gain: float,
) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="oshea2018_tx_", suffix=".cs8", delete=False) as f:
            tmp_path = Path(f.name)
            complex64_to_cs8(iq).tofile(f)
        amp_enable = 1 if amp_gain > 0 else 0
        command = [
            "hackrf_transfer",
            "-t",
            str(tmp_path),
            "-f",
            str(int(round(center_freq))),
            "-s",
            str(int(round(sample_rate))),
            "-x",
            str(int(round(vga_gain))),
            "-a",
            str(amp_enable),
            "-n",
            str(max(1, int(round(sample_rate * seconds)))),
            "-R",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(f"hackrf_transfer TX failed: {result.returncode}")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def transmit_with_gnuradio(
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    seconds: float,
    amp_gain: float,
    vga_gain: float,
) -> None:
    try:
        from gnuradio import blocks, gr
        import osmosdr
    except Exception as exc:
        raise SystemExit(f"GNU Radio/osmosdr TX backend is unavailable: {exc}") from exc

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="oshea2018_tx_", suffix=".complex64", delete=False) as f:
            tmp_path = Path(f.name)
            iq.tofile(f)

        tb = gr.top_block("oshea2018_hackrf_tx")
        source = blocks.file_source(gr.sizeof_gr_complex, str(tmp_path), True)
        throttle = blocks.throttle(gr.sizeof_gr_complex, sample_rate, True)
        sink = osmosdr.sink(args="numchan=1 hackrf=0")
        sink.set_sample_rate(sample_rate)
        sink.set_center_freq(center_freq, 0)
        sink.set_gain(vga_gain, 0)
        try:
            sink.set_if_gain(amp_gain, 0)
        except Exception:
            pass
        try:
            sink.set_bb_gain(vga_gain, 0)
        except Exception:
            pass
        sink.set_bandwidth(0, 0)
        tb.connect(source, throttle, sink)
        tb.start()
        time.sleep(max(0.1, seconds))
        tb.stop()
        tb.wait()
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulation", choices=["BASK", "BFSK", "BPSK"], required=True)
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--baseband-offset-hz", type=float, default=None)
    parser.add_argument("--backend", choices=["hackrf_transfer", "gnuradio", "soapy"], default="hackrf_transfer")
    args = parser.parse_args()
    transmit_oshea2018(
        args.modulation,
        args.config,
        args.seconds,
        args.seed,
        args.tx_vga_gain,
        args.tx_amp_gain,
        args.baseband_offset_hz,
        args.backend,
    )


if __name__ == "__main__":
    main()
