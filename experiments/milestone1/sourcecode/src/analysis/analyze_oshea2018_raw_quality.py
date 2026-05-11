from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.import_oshea2018_ota_windows import capture_quality, detect_active_region, window_region


def analyze_raw_quality(config_path: str, raw_root: str | None, output_dir: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(raw_root or cfg["dataset"]["raw_ota_root"])
    sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    rx_lead = float(cfg["ota"].get("rx_lead_seconds", 0.5))
    guard = float(cfg["ota"].get("tx_guard_seconds", 0.2))
    window_len = int(cfg["dataset"].get("window_len", 1024))
    active_cfg = cfg["ota"].get("active_region_detection", {})
    quality_cfg = cfg["ota"].get("quality_filter", {})
    pre_range = (0.0, min(0.2, rx_lead))
    tx_range = (rx_lead + guard, min(float(cfg["ota"].get("capture_seconds", 5.0)) - guard, rx_lead + guard + 1.0))
    late_range = (min(2.0, tx_range[1]), min(3.0, float(cfg["ota"].get("capture_seconds", 5.0)) - guard))

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("session_*/*.bin")):
        modulation = parse_modulation(path)
        if modulation is None:
            continue
        iq = np.fromfile(path, dtype=np.complex64)
        fallback_start, fallback_end = window_region(
            total_samples=len(iq),
            sample_rate=sample_rate,
            window_len=window_len,
            start_seconds=float(cfg["ota"].get("window_start_seconds", rx_lead + guard)),
            end_seconds=None,
            guard_seconds=guard,
        )
        detection = detect_active_region(iq, sample_rate, window_len, rx_lead, fallback_start, fallback_end, active_cfg)
        start = int(detection["active_start_sample"]) if detection["active_region_found"] else fallback_start
        end = int(detection["active_end_sample"]) if detection["active_region_found"] else fallback_end
        quality = capture_quality(iq, sample_rate, start, end, rx_lead, quality_cfg)
        pre = segment_rms(iq, sample_rate, *pre_range)
        tx = segment_rms(iq, sample_rate, *tx_range)
        late = segment_rms(iq, sample_rate, *late_range)
        rows.append(
            {
                "file": str(path),
                "session_id": path.parent.name,
                "modulation": modulation,
                "pre_rms": pre,
                "tx_rms": tx,
                "late_rms": late,
                "tx_to_pre_rms_ratio": float(tx / max(pre, 1e-12)),
                "late_to_pre_rms_ratio": float(late / max(pre, 1e-12)),
                "active_region_found": bool(detection["active_region_found"]),
                "active_start_seconds": float(start / sample_rate),
                "active_end_seconds": float(end / sample_rate),
                "quality_pass": bool(quality["quality_pass"]),
                "quality_reason": str(quality["quality_reason"]),
                "quality_tx_to_noise_rms_ratio": float(quality["tx_to_noise_rms_ratio"]),
                "spectral_peak_prominence": float(quality["spectral_peak_prominence"]),
                "clipping_rate": float(quality["clipping_rate"]),
            }
        )
    summary = summarize(rows)
    result = {
        "raw_root": str(root),
        "ranges_seconds": {"pre": pre_range, "tx": tx_range, "late": late_range},
        "summary_by_modulation": summary,
        "rows": rows,
    }
    out = ensure_dir(output_dir)
    write_json(out / "raw_quality_summary.json", result)
    (out / "raw_quality_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    return result


def parse_modulation(path: Path) -> str | None:
    lower = path.name.lower()
    for name in CLASS_NAMES:
        if name.lower() in lower:
            return name
    return None


def segment_rms(iq: np.ndarray, sample_rate: float, start_seconds: float, end_seconds: float) -> float:
    start = max(0, int(round(start_seconds * sample_rate)))
    end = min(len(iq), int(round(end_seconds * sample_rate)))
    if end <= start:
        return 0.0
    segment = iq[start:end]
    return float(np.sqrt(np.mean(np.abs(segment) ** 2)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for modulation in CLASS_NAMES:
        items = [row for row in rows if row["modulation"] == modulation]
        if not items:
            continue
        tx_ratio = np.asarray([row["tx_to_pre_rms_ratio"] for row in items], dtype=np.float64)
        quality_ratio = np.asarray([row["quality_tx_to_noise_rms_ratio"] for row in items], dtype=np.float64)
        late_ratio = np.asarray([row["late_to_pre_rms_ratio"] for row in items], dtype=np.float64)
        tx_rms = np.asarray([row["tx_rms"] for row in items], dtype=np.float64)
        passed = np.asarray([row["quality_pass"] for row in items], dtype=np.float64)
        active_found = np.asarray([row["active_region_found"] for row in items], dtype=np.float64)
        summary[modulation] = {
            "captures": float(len(items)),
            "tx_rms_median": float(np.median(tx_rms)),
            "tx_to_pre_ratio_median": float(np.median(tx_ratio)),
            "tx_to_pre_ratio_p10": float(np.percentile(tx_ratio, 10)),
            "tx_to_pre_ratio_p90": float(np.percentile(tx_ratio, 90)),
            "quality_tx_to_noise_ratio_median": float(np.median(quality_ratio)),
            "active_region_found_rate": float(np.mean(active_found)),
            "quality_pass_rate": float(np.mean(passed)),
            "late_to_pre_ratio_median": float(np.median(late_ratio)),
            "weak_capture_rate_ratio_lt_1_5": float(np.mean(tx_ratio < 1.5)),
        }
    return summary


def render_markdown(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Oshea2018 Raw OTA Quality",
        "",
        "| modulation | captures | tx_rms_median | tx/pre median | tx/pre p10 | tx/pre p90 | weak/pass rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for modulation in CLASS_NAMES:
        row = summary.get(modulation)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    modulation,
                    f"{int(row['captures'])}",
                    f"{row['tx_rms_median']:.6f}",
                    f"{row['tx_to_pre_ratio_median']:.3f}",
                    f"{row['tx_to_pre_ratio_p10']:.3f}",
                    f"{row['tx_to_pre_ratio_p90']:.3f}",
                    f"{row['weak_capture_rate_ratio_lt_1_5']:.3f} / pass {row['quality_pass_rate']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-dir", default="../results/raw_quality")
    args = parser.parse_args()
    analyze_raw_quality(args.config, args.raw_root, args.output_dir)


if __name__ == "__main__":
    main()
