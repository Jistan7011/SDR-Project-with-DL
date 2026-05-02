from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter
from tqdm import tqdm

from src.common import ensure_dir, load_config, write_json
from src.signal.frame import make_frame
from src.signal.processing import iq_to_channels


def import_real_iq(
    input_path: str,
    modulation: str,
    payload: str,
    config_path: str = "config.yaml",
    output_root: str = "data/real/processed",
    split: str = "test",
    skip_seconds: float = 0.0,
    duration_seconds: float | None = None,
    stride: int | None = None,
    max_windows: int = 300,
    freq_shift_hz: float = 0.0,
    channel_center_hz: float | None = None,
    channel_bandwidth_hz: float | None = None,
    filter_taps: int = 129,
) -> Path:
    cfg = load_config(config_path)
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    window_size = int(cfg["dataset"]["window_size"])
    stride = stride or window_size
    raw_path = Path(input_path)
    iq = np.fromfile(raw_path, dtype=np.complex64)
    start = int(skip_seconds * sample_rate)
    end = len(iq) if duration_seconds is None else min(len(iq), start + int(duration_seconds * sample_rate))
    iq = iq[start:end]
    if freq_shift_hz:
        t = np.arange(len(iq), dtype=np.float64) / sample_rate
        iq = (iq * np.exp(-1j * 2.0 * np.pi * freq_shift_hz * t)).astype(np.complex64)
    if channel_center_hz is not None and channel_bandwidth_hz is not None:
        iq = channel_filter(
            iq,
            sample_rate=sample_rate,
            channel_center_hz=channel_center_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            target_center_hz=100_000.0,
            taps=filter_taps,
        )
    if len(iq) < window_size:
        raise SystemExit(f"Not enough samples after trimming: {len(iq)} < {window_size}")

    out_dir = ensure_dir(Path(output_root) / split)
    bits = make_frame(payload)
    count = min(max_windows, 1 + (len(iq) - window_size) // stride)
    manifest: list[dict[str, object]] = []
    stem = raw_path.stem
    for idx in tqdm(range(count), desc=f"import:{stem}"):
        offset = idx * stride
        window = iq[offset : offset + window_size]
        out_path = out_dir / f"{stem}_{idx:06d}.npz"
        np.savez(
            out_path,
            iq=iq_to_channels(window, window_size),
            raw_iq=window.astype(np.complex64),
            modulation=modulation.upper(),
            payload=payload,
            bits=bits.astype(np.uint8),
            snr_db=np.nan,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            center_freq=int(cfg["sdr"]["center_freq"]),
            session_id=stem,
            source_file=str(raw_path),
            source_offset=int(start + offset),
        )
        manifest.append({"path": str(out_path), "offset": int(start + offset), "modulation": modulation.upper(), "payload": payload})

    meta_dir = ensure_dir(Path(output_root).parent / "metadata")
    write_json(
        meta_dir / f"{stem}.json",
        {
            "source_file": str(raw_path),
            "modulation": modulation.upper(),
            "payload": payload,
            "center_freq": int(cfg["sdr"]["center_freq"]),
            "tx_sample_rate": float(cfg["sdr"]["tx_sample_rate"]),
            "rx_sample_rate": sample_rate,
            "symbol_rate": symbol_rate,
            "tx_gain": float(cfg["sdr"]["tx_gain"]),
            "rx_gain": float(cfg["sdr"]["rx_gain"]),
            "attenuator": "30~60 dB",
            "skip_seconds": skip_seconds,
            "duration_seconds": duration_seconds,
            "window_size": window_size,
            "stride": stride,
            "windows": count,
            "freq_shift_hz": freq_shift_hz,
            "channel_center_hz": channel_center_hz,
            "channel_bandwidth_hz": channel_bandwidth_hz,
            "filter_taps": filter_taps,
        },
    )
    write_json(Path(output_root) / f"manifest_{stem}.json", {"samples": manifest})
    print(f"Imported {count} windows to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--modulation", choices=["BASK", "BFSK", "BPSK"], required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="data/real/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--freq-shift-hz", type=float, default=0.0, help="Mix this frequency offset down to baseband before windowing.")
    parser.add_argument("--channel-center-hz", type=float, default=None, help="Band-limit around this captured frequency offset, then place it at +100 kHz.")
    parser.add_argument("--channel-bandwidth-hz", type=float, default=None)
    parser.add_argument("--filter-taps", type=int, default=129)
    args = parser.parse_args()
    import_real_iq(
        input_path=args.input,
        modulation=args.modulation,
        payload=args.payload,
        config_path=args.config,
        output_root=args.output_root,
        split=args.split,
        skip_seconds=args.skip_seconds,
        duration_seconds=args.duration_seconds,
        stride=args.stride,
        max_windows=args.max_windows,
        freq_shift_hz=args.freq_shift_hz,
        channel_center_hz=args.channel_center_hz,
        channel_bandwidth_hz=args.channel_bandwidth_hz,
        filter_taps=args.filter_taps,
    )


def channel_filter(
    iq: np.ndarray,
    sample_rate: float,
    channel_center_hz: float,
    channel_bandwidth_hz: float,
    target_center_hz: float,
    taps: int,
) -> np.ndarray:
    t = np.arange(len(iq), dtype=np.float64) / sample_rate
    shifted = iq * np.exp(-1j * 2.0 * np.pi * channel_center_hz * t)
    cutoff = min(channel_bandwidth_hz / 2.0, sample_rate * 0.45)
    fir = firwin(taps, cutoff=cutoff, fs=sample_rate)
    filtered = lfilter(fir, [1.0], shifted)
    restored = filtered * np.exp(1j * 2.0 * np.pi * target_center_hz * t)
    return restored.astype(np.complex64)


if __name__ == "__main__":
    main()
