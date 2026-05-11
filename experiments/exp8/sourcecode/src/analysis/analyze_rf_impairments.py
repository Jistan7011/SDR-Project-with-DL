from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common import ensure_dir, load_config, write_json
from src.signal.channelize import estimate_snr_db, spectrum_summary


EPS = 1e-12


def resolve_capture_path(value: str | Path, session_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    local_path = session_dir / path.name
    if local_path.exists():
        return local_path
    raise FileNotFoundError(f"Cannot resolve capture path: {value}")


def read_complex_segment(path: Path, start: int = 0, count: int | None = None) -> np.ndarray:
    data = np.memmap(path, dtype=np.complex64, mode="r")
    start = max(0, min(int(start), len(data)))
    end = len(data) if count is None else min(len(data), start + int(count))
    return np.asarray(data[start:end], dtype=np.complex64)


def spectrum_peak(iq: np.ndarray, sample_rate: float) -> dict[str, float]:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) == 0:
        return {"peak_frequency_hz": float("nan"), "peak_db": float("nan"), "noise_floor_db": float("nan")}
    n = min(262_144, len(arr))
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(arr[:n] * window))
    power_db = 20.0 * np.log10(np.abs(spectrum) + EPS)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(sample_rate)))
    idx = int(np.argmax(power_db))
    return {
        "peak_frequency_hz": float(freqs[idx]),
        "peak_db": float(power_db[idx]),
        "noise_floor_db": float(np.median(power_db)),
    }


def spectral_flatness(iq: np.ndarray, sample_rate: float, center_hz: float) -> float:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) == 0:
        return float("nan")
    n = min(262_144, len(arr))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    shifted = arr[:n] * np.exp(-1j * 2.0 * np.pi * float(center_hz) * t)
    spectrum = np.fft.fftshift(np.fft.fft(shifted * np.hanning(n)))
    power = np.abs(spectrum) ** 2 + EPS
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def iq_hardware_metrics(iq: np.ndarray) -> dict[str, float]:
    arr = np.asarray(iq, dtype=np.complex64)
    if len(arr) == 0:
        return {}
    i = arr.real.astype(np.float64)
    q = arr.imag.astype(np.float64)
    mean_complex = complex(float(np.mean(i)), float(np.mean(q)))
    rms = float(np.sqrt(np.mean(np.abs(arr) ** 2)))
    i_power = float(np.mean(i * i))
    q_power = float(np.mean(q * q))
    i_centered = i - float(np.mean(i))
    q_centered = q - float(np.mean(q))
    corr = float(np.mean(i_centered * q_centered) / (np.std(i_centered) * np.std(q_centered) + EPS))
    corr = max(-1.0, min(1.0, corr))
    return {
        "rms": rms,
        "rms_db": 20.0 * float(np.log10(rms + EPS)),
        "dc_i": float(mean_complex.real),
        "dc_q": float(mean_complex.imag),
        "dc_magnitude": float(abs(mean_complex)),
        "dc_to_rms_db": 20.0 * float(np.log10((abs(mean_complex) + EPS) / (rms + EPS))),
        "i_power": i_power,
        "q_power": q_power,
        "iq_power_ratio_db": 10.0 * float(np.log10((i_power + EPS) / (q_power + EPS))),
        "iq_correlation": corr,
        "quadrature_error_proxy_deg": float(np.degrees(np.arcsin(corr))),
    }


def analyze_capture(
    session_dir: Path,
    metadata: dict[str, Any],
    capture: dict[str, Any],
    active_start_seconds: float,
    active_duration_seconds: float,
    fft_samples: int,
) -> dict[str, Any]:
    sample_rate = float(metadata["rx_sample_rate"])
    baseband_offset_hz = float(metadata.get("baseband_offset_hz", 0.0))
    source = resolve_capture_path(capture["file"], session_dir)
    noise_path = resolve_capture_path(metadata["noise_only_file"], session_dir)
    active_start = int(active_start_seconds * sample_rate)
    active_count = int(active_duration_seconds * sample_rate)
    active = read_complex_segment(source, active_start, min(active_count, fft_samples))
    noise = read_complex_segment(noise_path, 0, min(int(float(metadata.get("noise_seconds", 3.0)) * sample_rate), fft_samples))

    active_metrics = iq_hardware_metrics(active)
    peak = spectrum_peak(active, sample_rate)
    noise_summary = spectrum_summary(noise, sample_rate)
    snr = estimate_snr_db(noise, active)
    flatness = spectral_flatness(active, sample_rate, baseband_offset_hz)

    return {
        "session_id": metadata.get("session_id", session_dir.name),
        "modulation": str(capture["modulation"]),
        "payload": str(capture["payload"]),
        "file": str(source),
        "tx_rx_distance_m": metadata.get("tx_rx_distance_m"),
        "antenna_layout": metadata.get("antenna_layout"),
        "tx_vga_gain": metadata.get("tx_vga_gain"),
        "rx_gain": metadata.get("rx_gain"),
        "baseband_offset_hz": baseband_offset_hz,
        "estimated_snr_db": snr["estimated_snr_db"],
        "noise_power": snr["noise_power"],
        "active_power": snr["active_power"],
        "active_peak_frequency_hz": peak["peak_frequency_hz"],
        "spectral_peak_residual_hz": peak["peak_frequency_hz"] - baseband_offset_hz,
        "active_peak_db": peak["peak_db"],
        "active_noise_floor_db": peak["noise_floor_db"],
        "noise_peak_frequency_hz": noise_summary["peak_frequency_hz"],
        "noise_floor_db": noise_summary["noise_floor_db"],
        "noise_interference_flag": noise_summary["interference_flag"],
        "spectral_flatness_proxy": flatness,
        **active_metrics,
    }


def aggregate_session(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "estimated_snr_db",
        "rms_db",
        "dc_to_rms_db",
        "iq_power_ratio_db",
        "iq_correlation",
        "quadrature_error_proxy_deg",
        "spectral_peak_residual_hz",
        "spectral_flatness_proxy",
    ]
    out: dict[str, Any] = {
        "session_id": rows[0]["session_id"],
        "captures": len(rows),
        "tx_rx_distance_m": rows[0].get("tx_rx_distance_m"),
        "antenna_layout": rows[0].get("antenna_layout"),
        "tx_vga_gain": rows[0].get("tx_vga_gain"),
        "rx_gain": rows[0].get("rx_gain"),
        "baseband_offset_hz": rows[0].get("baseband_offset_hz"),
    }
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))], dtype=np.float64)
        if len(values) == 0:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
        else:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(session_rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Experiment 3 RF Impairment Audit",
        "",
        "## 판단",
        "",
        "팀원 메모에서 실험 3에 채택할 수 있는 부분은 실제 capture에서 측정 가능한 RF impairment를 기록하는 것이다. Synthetic pretraining 후 fine-tuning은 일반적으로 타당한 전략이지만, 현재 실험 3은 이미 OTA 데이터 기반 학습/평가이므로 즉시 핵심 절차로 채택하지 않는다.",
        "",
        "## 채택한 항목",
        "",
        "- gain variation: TX/RX gain 조건과 capture RMS로 기록",
        "- carrier/LO offset: active spectrum peak와 계획된 baseband offset의 차이를 proxy로 기록",
        "- phase offset: 절대 위상은 랜덤으로 보고 직접 보정하지 않음",
        "- DC offset: raw active IQ 평균과 RMS 대비 DC 비율 기록",
        "- IQ imbalance: I/Q power ratio와 I-Q correlation을 proxy로 기록",
        "- multipath fading: OTA 환경 요인으로 인정하되, 단일 수신기 capture에서는 직접 추정하지 않고 spectral flatness proxy만 기록",
        "",
        "## 채택하지 않은 항목",
        "",
        "- USRP 표현: 현재 장비는 HackRF One + RTL-SDR V4이므로 문서에는 USRP라고 쓰지 않음",
        "- sample-rate offset 직접 추정: 현재 frame/capture 구조만으로 신뢰도 있게 추정하기 어려워 audit 지표에서 제외",
        "- early layer freeze fine-tuning: 타당한 후보지만 현재 exp3의 1차 개선은 OTA impairment audit과 feature 선택이며, synthetic-to-OTA transfer 실험은 별도 실험으로 분리",
        "",
        "## Session Summary",
        "",
        "| session | snr mean | rms dB mean | DC/RMS dB mean | IQ power ratio dB | IQ corr | peak residual Hz | flatness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in session_rows:
        lines.append(
            "| {session_id} | {estimated_snr_db_mean:.2f} | {rms_db_mean:.2f} | {dc_to_rms_db_mean:.2f} | {iq_power_ratio_db_mean:.2f} | {iq_correlation_mean:.3f} | {spectral_peak_residual_hz_mean:.1f} | {spectral_flatness_proxy_mean:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 실험 3 개선 결론",
            "",
            "실험 3의 다음 개선은 synthetic fine-tuning보다 먼저, 실제 OTA capture에서 관측된 DC/IQ/CFO/gain 분포를 augmentation 또는 preprocessing 범위로 반영하는 것이다. 특히 현재 best balanced model이 `[I,Q,instantaneous_frequency]`인 점을 고려하면, carrier/phase/frequency 계열 feature가 중요한 것으로 판단한다.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_rf_impairments(
    raw_root: str = "../data/raw_iq",
    config_path: str = "../config/config.exp03.yaml",
    output_dir: str = "../results/reports/rf_impairments",
    session_start: int = 16,
    session_end: int = 21,
    fft_samples: int = 262_144,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    import_cfg = cfg.get("experiment3", {}).get("import", {})
    active_start_seconds = float(import_cfg.get("active_start_seconds", 1.1))
    active_duration_seconds = float(import_cfg.get("active_duration_seconds", 4.5))
    root = Path(raw_root)
    out = ensure_dir(output_dir)
    capture_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for session_num in range(session_start, session_end + 1):
        session_dir = root / f"session_{session_num:03d}"
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = [
            analyze_capture(session_dir, metadata, capture, active_start_seconds, active_duration_seconds, fft_samples)
            for capture in metadata.get("captures", [])
        ]
        capture_rows.extend(rows)
        session_rows.append(aggregate_session(rows))

    result = {"captures": capture_rows, "sessions": session_rows}
    write_json(out / "rf_impairment_audit.json", result)
    write_csv(out / "rf_impairment_capture_metrics.csv", capture_rows)
    write_csv(out / "rf_impairment_session_summary.csv", session_rows)
    make_markdown(session_rows, out / "RF_IMPAIRMENT_AUDIT.md")
    print(f"RF impairment audit written to {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="../data/raw_iq")
    parser.add_argument("--config", default="../config/config.exp03.yaml")
    parser.add_argument("--output-dir", default="../results/reports/rf_impairments")
    parser.add_argument("--session-start", type=int, default=16)
    parser.add_argument("--session-end", type=int, default=21)
    parser.add_argument("--fft-samples", type=int, default=262_144)
    args = parser.parse_args()
    analyze_rf_impairments(args.raw_root, args.config, args.output_dir, args.session_start, args.session_end, args.fft_samples)


if __name__ == "__main__":
    main()
