from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.import_oshea2018_ota_windows import capture_quality_against_noise
from src.signal.channelize import channelize_and_downsample, estimate_snr_db, spectrum_summary


def load_metadata(session_dir: Path) -> dict[str, object]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def active_region(iq: np.ndarray, sample_rate: float, cfg: dict) -> np.ndarray:
    start = int(round(float(cfg["ota"].get("active_start_seconds", 1.1)) * sample_rate))
    duration = float(cfg["ota"].get("active_duration_seconds", 3.8))
    end = min(len(iq), start + int(round(duration * sample_rate)))
    return iq[start:end]


def feature_stats(iq: np.ndarray, sample_rate: float, symbol_rate: float) -> dict[str, float]:
    if len(iq) == 0:
        return {
            "magnitude_cv": 0.0,
            "ifreq_std": 0.0,
            "dphase_abs_mean": 0.0,
            "dphase_abs_p95": 0.0,
        }
    centered = iq.astype(np.complex64) - np.mean(iq.astype(np.complex64))
    rms = np.sqrt(np.mean(np.abs(centered) ** 2)) + 1e-8
    norm = centered / rms
    magnitude = np.abs(norm)
    if len(norm) > 1:
        ifreq = np.angle(norm[1:] * np.conj(norm[:-1]))
    else:
        ifreq = np.zeros(1, dtype=np.float32)
    delay = max(1, int(round(sample_rate / symbol_rate)))
    if len(norm) > delay:
        dphase = np.angle(norm[delay:] * np.conj(norm[:-delay]))
    else:
        dphase = np.zeros(1, dtype=np.float32)
    return {
        "magnitude_cv": float(np.std(magnitude) / (np.mean(magnitude) + 1e-8)),
        "ifreq_std": float(np.std(ifreq)),
        "dphase_abs_mean": float(np.mean(np.abs(dphase))),
        "dphase_abs_p95": float(np.percentile(np.abs(dphase), 95)),
    }


def find_capture_meta(metadata: dict[str, object], path: Path) -> dict[str, object]:
    for item in metadata.get("captures", []):
        if Path(str(item.get("file", ""))).name == path.name:
            return dict(item)
    return {}


def condition_key(row: dict[str, object]) -> str:
    return f"tx{row['tx_vga_gain']:g}_rx{row['rx_gain']:g}_off{int(row['baseband_offset_hz'])}"


def pct(items: list[bool]) -> float:
    return float(np.mean(items)) if items else 0.0


def med(items: list[float]) -> float:
    clean = [float(item) for item in items if np.isfinite(float(item))]
    return float(median(clean)) if clean else 0.0


def summarize_condition(rows: list[dict[str, object]]) -> dict[str, object]:
    by_mod: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_mod[str(row["modulation"])].append(row)
    per_mod: dict[str, dict[str, float | int]] = {}
    for mod in CLASS_NAMES:
        mod_rows = by_mod.get(mod, [])
        per_mod[mod] = {
            "captures": len(mod_rows),
            "pass_rate": pct([bool(row["quality_pass"]) for row in mod_rows]),
            "tx_to_noise_rms_median": med([float(row["tx_to_noise_rms_ratio"]) for row in mod_rows]),
            "estimated_snr_db_median": med([float(row["estimated_snr_db"]) for row in mod_rows]),
            "clipping_rate_median": med([float(row["clipping_rate"]) for row in mod_rows]),
            "dphase_abs_p95_median": med([float(row["dphase_abs_p95"]) for row in mod_rows]),
            "dphase_abs_mean_median": med([float(row["dphase_abs_mean"]) for row in mod_rows]),
            "magnitude_cv_median": med([float(row["magnitude_cv"]) for row in mod_rows]),
        }
    bpsk_sep = float(per_mod["BPSK"]["dphase_abs_p95_median"]) - float(per_mod["BASK"]["dphase_abs_p95_median"])
    balanced_pass = min(float(per_mod[mod]["pass_rate"]) for mod in CLASS_NAMES)
    bpsk_bask_pass = min(float(per_mod["BASK"]["pass_rate"]), float(per_mod["BPSK"]["pass_rate"]))
    score = balanced_pass * 2.0 + bpsk_bask_pass * 2.0 + max(0.0, bpsk_sep)
    return {
        "condition": condition_key(rows[0]),
        "tx_vga_gain": rows[0]["tx_vga_gain"],
        "tx_amp_gain": rows[0]["tx_amp_gain"],
        "rx_gain": rows[0]["rx_gain"],
        "baseband_offset_hz": rows[0]["baseband_offset_hz"],
        "per_modulation": per_mod,
        "bpsk_minus_bask_dphase_abs_p95": bpsk_sep,
        "balanced_pass_rate": balanced_pass,
        "bask_bpsk_pass_rate": bpsk_bask_pass,
        "recommendation_score": score,
        "recommend_for_full_collection": bool(bpsk_bask_pass >= 0.8 and balanced_pass >= 0.8),
    }


def analyze_preflight(config_path: str, raw_root: str, output_dir: str) -> dict[str, object]:
    cfg = load_config(config_path)
    raw = Path(raw_root)
    out = ensure_dir(output_dir)
    raw_sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    target_sample_rate = float(cfg["dataset"].get("target_sample_rate", raw_sample_rate))
    channel_bandwidth_hz = float(cfg["dataset"].get("channel_bandwidth_hz", 100_000.0))
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    rows: list[dict[str, object]] = []

    for session_dir in sorted(path for path in raw.iterdir() if path.is_dir()):
        metadata = load_metadata(session_dir)
        if not metadata:
            continue
        noise_path = Path(str(metadata.get("noise_only_file", session_dir / "noise_only.bin")))
        if not noise_path.exists():
            noise_path = session_dir / "noise_only.bin"
        if not noise_path.exists():
            continue
        noise_raw = np.fromfile(noise_path, dtype=np.complex64)
        baseband_offset_hz = float(metadata.get("baseband_offset_hz", cfg["sdr"].get("baseband_offset_hz", 500_000.0)))
        noise_channelized, effective_sample_rate = channelize_and_downsample(
            noise_raw,
            sample_rate=raw_sample_rate,
            channel_center_hz=baseband_offset_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            target_sample_rate=target_sample_rate,
        )
        for path in sorted(session_dir.glob("*.bin")):
            if path.name == "noise_only.bin":
                continue
            modulation = next((name for name in CLASS_NAMES if name.lower() in path.name.lower()), None)
            if modulation is None:
                continue
            capture_raw = np.fromfile(path, dtype=np.complex64)
            active_raw = active_region(capture_raw, raw_sample_rate, cfg)
            active_channelized, effective_sample_rate = channelize_and_downsample(
                active_raw,
                sample_rate=raw_sample_rate,
                channel_center_hz=baseband_offset_hz,
                channel_bandwidth_hz=channel_bandwidth_hz,
                target_sample_rate=target_sample_rate,
            )
            quality = capture_quality_against_noise(
                noise_channelized,
                active_channelized,
                effective_sample_rate,
                cfg["ota"].get("quality_filter", {}),
            )
            snr = estimate_snr_db(noise_channelized, active_channelized)
            spec = spectrum_summary(active_channelized, effective_sample_rate)
            feats = feature_stats(active_channelized, effective_sample_rate, symbol_rate)
            cap_meta = find_capture_meta(metadata, path)
            row = {
                "session_id": session_dir.name,
                "file": str(path),
                "modulation": modulation,
                "tx_vga_gain": float(metadata.get("tx_vga_gain", 0.0)),
                "tx_amp_gain": float(metadata.get("tx_amp_gain", 0.0)),
                "rx_gain": float(metadata.get("rx_gain", 0.0)),
                "baseband_offset_hz": float(baseband_offset_hz),
                "payload_seed": int(cap_meta.get("payload_seed", -1)),
                "random_payload_bits": bool(cap_meta.get("random_payload_bits", metadata.get("random_payload_bits", True))),
                **quality,
                **snr,
                **spec,
                **feats,
            }
            rows.append(row)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)
    conditions = [summarize_condition(grouped[key]) for key in sorted(grouped)]
    conditions_sorted = sorted(conditions, key=lambda item: float(item["recommendation_score"]), reverse=True)
    result = {
        "config": str(config_path),
        "raw_root": str(raw),
        "capture_count": len(rows),
        "conditions": conditions_sorted,
        "best_condition": conditions_sorted[0] if conditions_sorted else None,
        "rows": rows,
    }
    write_json(out / "fixed_preflight_quality.json", result)
    write_markdown(out / "fixed_preflight_summary.md", result)
    print(json.dumps({k: result[k] for k in ("capture_count", "best_condition")}, indent=2, ensure_ascii=False))
    return result


def write_markdown(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Oshea2018 Fixed Preflight Summary",
        "",
        f"- capture_count: `{result['capture_count']}`",
        "",
        "| condition | BASK pass | BFSK pass | BPSK pass | BPSK-BASK dphase p95 | score | full collection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cond in result["conditions"]:
        per = cond["per_modulation"]
        lines.append(
            "| {condition} | {bask:.3f} | {bfsk:.3f} | {bpsk:.3f} | {sep:.4f} | {score:.4f} | {go} |".format(
                condition=cond["condition"],
                bask=per["BASK"]["pass_rate"],
                bfsk=per["BFSK"]["pass_rate"],
                bpsk=per["BPSK"]["pass_rate"],
                sep=cond["bpsk_minus_bask_dphase_abs_p95"],
                score=cond["recommendation_score"],
                go="PASS" if cond["recommend_for_full_collection"] else "HOLD",
            )
        )
    lines.extend(["", "## Per Condition Details", ""])
    for cond in result["conditions"]:
        lines.append(f"### {cond['condition']}")
        lines.append("")
        lines.append("| modulation | captures | pass | tx/noise median | snr median | clip median | dphase p95 median | mag cv median |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mod, item in cond["per_modulation"].items():
            lines.append(
                "| {mod} | {captures} | {pass_rate:.3f} | {ratio:.3f} | {snr:.2f} | {clip:.4f} | {dphase:.4f} | {mag:.4f} |".format(
                    mod=mod,
                    captures=item["captures"],
                    pass_rate=item["pass_rate"],
                    ratio=item["tx_to_noise_rms_median"],
                    snr=item["estimated_snr_db_median"],
                    clip=item["clipping_rate_median"],
                    dphase=item["dphase_abs_p95_median"],
                    mag=item["magnitude_cv_median"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.fixed.yaml")
    parser.add_argument("--raw-root", default="../data/raw_ota_fixed_preflight")
    parser.add_argument("--output-dir", default="../results/fixed_preflight_analysis")
    args = parser.parse_args()
    analyze_preflight(args.config, args.raw_root, args.output_dir)


if __name__ == "__main__":
    main()
