from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.signal.channelize import channelize_and_downsample, estimate_snr_db, spectrum_summary
from src.signal.oshea2018_waveform import to_unit_variance_iq


MANIFEST_PREVIEW_LIMIT = 1000


def split_for_counter(counter: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cut = int(round(total * train_ratio))
    val_cut = train_cut + int(round(total * val_ratio))
    if counter < train_cut:
        return "train"
    if counter < val_cut:
        return "val"
    return "test"


def parse_modulation_from_name(path: Path) -> str:
    lower = path.name.lower()
    for name in CLASS_NAMES:
        if name.lower() in lower:
            return name
    raise ValueError(f"Could not infer modulation from filename: {path}")


def import_oshea2018_ota_windows(
    config_path: str,
    raw_root: str | None = None,
    output_root: str | None = None,
    clean: bool = False,
) -> Path:
    cfg = load_config(config_path)
    raw = Path(raw_root or cfg["dataset"]["raw_ota_root"])
    out = Path(output_root or cfg["dataset"].get("ota_clean_root", cfg["dataset"]["ota_root"]))
    if clean and out.exists():
        shutil.rmtree(out)
    ensure_dir(out)
    for split in ("train", "val", "test"):
        ensure_dir(out / split)
    window_len = int(cfg["dataset"]["window_len"])
    stride = int(cfg["dataset"]["stride"])
    examples_per_capture_limit = int(cfg["ota"].get("examples_per_capture_limit", 0))
    raw_sample_rate = float(cfg["sdr"]["rx_sample_rate"])
    target_sample_rate = float(cfg["dataset"].get("target_sample_rate", raw_sample_rate))
    channel_bandwidth_hz = float(cfg["dataset"].get("channel_bandwidth_hz", 100_000.0))
    default_baseband_offset_hz = float(cfg["sdr"].get("baseband_offset_hz", 500_000.0))
    symbol_rate = float(cfg["sdr"]["symbol_rate"])
    rx_lead_seconds = float(cfg["ota"].get("rx_lead_seconds", 0.0))
    tx_guard_seconds = float(cfg["ota"].get("tx_guard_seconds", 0.2))
    window_start_seconds = float(cfg["ota"].get("window_start_seconds", rx_lead_seconds + tx_guard_seconds))
    window_end_seconds = cfg["ota"].get("window_end_seconds", None)
    sampling_mode = str(cfg["ota"].get("window_sampling", "uniform_tx_region")).lower().replace("-", "_")
    active_cfg = cfg["ota"].get("active_region_detection", {})
    quality_cfg = cfg["ota"].get("quality_filter", {})
    active_enabled = bool(active_cfg.get("enabled", True))
    quality_enabled = bool(quality_cfg.get("enabled", True))
    balance_after_import = bool(cfg["ota"].get("balance_after_import", True))
    train_ratio = float(cfg["dataset"]["train_ratio"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    bin_files = sorted(path for path in raw.glob("**/*.bin") if path.name.lower() != "noise_only.bin")
    if not bin_files:
        raise FileNotFoundError(f"No .bin OTA captures found in {raw}")
    session_ids = sorted({path.parent.name for path in bin_files})
    session_to_split = {
        session: split_for_counter(idx, len(session_ids), train_ratio, val_ratio)
        for idx, session in enumerate(session_ids)
    }
    counters = {"train": 0, "val": 0, "test": 0}
    manifest: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    manifest_csv = out / "samples_manifest.csv"
    manifest_csv.write_text(
        "path,split,modulation,session_id,source,source_start_seconds,active_start_seconds,active_end_seconds,source_rms,noise_rms,tx_rms,snr_like_ratio,spectral_peak_prominence,clipping_rate,quality_pass\n",
        encoding="utf-8",
    )
    for path in tqdm(bin_files, desc="import ota"):
        modulation = parse_modulation_from_name(path)
        session_id = path.parent.name
        split = session_to_split[session_id]
        raw_iq = np.fromfile(path, dtype=np.complex64)
        metadata = load_session_metadata(path.parent)
        capture_meta = load_capture_metadata(path.parent, path)
        baseband_offset_hz = float(capture_meta.get("baseband_offset_hz", metadata.get("baseband_offset_hz", default_baseband_offset_hz)))
        noise_path = Path(str(metadata.get("noise_only_file", path.parent / "noise_only.bin")))
        if not noise_path.exists():
            noise_path = path.parent / "noise_only.bin"
        noise_iq = np.fromfile(noise_path, dtype=np.complex64) if noise_path.exists() else raw_iq[: max(1, int(round(rx_lead_seconds * raw_sample_rate)))]
        fallback_start, fallback_end = window_region(
            total_samples=len(raw_iq),
            sample_rate=raw_sample_rate,
            window_len=window_len,
            start_seconds=float(cfg["ota"].get("active_start_seconds", window_start_seconds)),
            end_seconds=float(window_end_seconds) if window_end_seconds is not None else None,
            guard_seconds=tx_guard_seconds,
        )
        if "active_duration_seconds" in cfg["ota"]:
            fallback_end = min(
                len(raw_iq),
                fallback_start + int(round(float(cfg["ota"]["active_duration_seconds"]) * raw_sample_rate)),
            )
        detection = detect_active_region(raw_iq, raw_sample_rate, window_len, rx_lead_seconds, fallback_start, fallback_end, active_cfg)
        if active_enabled and detection["active_region_found"]:
            start_index = int(detection["active_start_sample"])
            end_index = int(detection["active_end_sample"])
        else:
            start_index, end_index = fallback_start, fallback_end
        active_raw = raw_iq[start_index:end_index]
        active_iq, effective_sample_rate = channelize_and_downsample(
            active_raw,
            sample_rate=raw_sample_rate,
            channel_center_hz=baseband_offset_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            target_sample_rate=target_sample_rate,
        )
        noise_channelized, _ = channelize_and_downsample(
            noise_iq,
            sample_rate=raw_sample_rate,
            channel_center_hz=baseband_offset_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            target_sample_rate=target_sample_rate,
        )
        quality = capture_quality_against_noise(noise_channelized, active_iq, effective_sample_rate, quality_cfg)
        snr_info = estimate_snr_db(noise_channelized, active_iq)
        spectrum_info = spectrum_summary(active_iq, effective_sample_rate)
        if quality_enabled and not bool(quality["quality_pass"]):
            rejected.append(
                {
                    "source": str(path),
                    "split": split,
                    "modulation": modulation,
                    "session_id": session_id,
                    "reason": quality["quality_reason"],
                    **detection,
                    **quality,
                    **snr_info,
                    **spectrum_info,
                }
            )
            continue
        indices = window_indices(
            start_index=0,
            end_index=len(active_iq),
            window_len=window_len,
            stride=stride,
            limit=examples_per_capture_limit,
            mode=sampling_mode,
        )
        produced = 0
        for index in indices:
            window = active_iq[index : index + window_len]
            channels = to_unit_variance_iq(window, window_len)
            name = f"{session_id}_{modulation.lower()}_{counters[split]:07d}.npz"
            out_path = out / split / name
            raw_index = start_index + int(round(index * raw_sample_rate / effective_sample_rate))
            sample_metadata = dict(metadata)
            for duplicated in ("baseband_offset_hz", "noise_only_file", "random_payload_bits"):
                sample_metadata.pop(duplicated, None)
            np.savez(
                out_path,
                iq=channels,
                raw_iq=window.astype(np.complex64),
                modulation=modulation,
                sample_rate=effective_sample_rate,
                raw_sample_rate=raw_sample_rate,
                symbol_rate=symbol_rate,
                session_id=session_id,
                source_capture=str(path),
                source_start_sample=int(raw_index),
                source_start_seconds=float(raw_index / raw_sample_rate),
                channelized_start_sample=int(index),
                channelized_start_seconds=float(index / effective_sample_rate),
                active_start_seconds=float(start_index / raw_sample_rate),
                active_end_seconds=float(end_index / raw_sample_rate),
                source_rms=window_rms(window),
                noise_rms=float(quality["noise_rms"]),
                tx_rms=float(quality["tx_rms"]),
                snr_like_ratio=float(quality["tx_to_noise_rms_ratio"]),
                estimated_snr_db=float(snr_info["estimated_snr_db"]),
                noise_power=float(snr_info["noise_power"]),
                active_power=float(snr_info["active_power"]),
                signal_power=float(snr_info["signal_power"]),
                spectral_peak_prominence=float(quality["spectral_peak_prominence"]),
                clipping_rate=float(quality["clipping_rate"]),
                quality_pass=bool(quality["quality_pass"]),
                baseband_offset_hz=baseband_offset_hz,
                channel_bandwidth_hz=channel_bandwidth_hz,
                target_sample_rate=target_sample_rate,
                noise_only_file=str(noise_path),
                payload_seed=int(capture_meta.get("payload_seed", -1)),
                random_payload_bits=bool(capture_meta.get("random_payload_bits", True)),
                **spectrum_info,
                dataset_kind="ota",
                experiment="Oshea2018",
                **sample_metadata,
            )
            row = {
                "path": str(out_path),
                "split": split,
                "modulation": modulation,
                "session_id": session_id,
                "source": str(path),
                "source_start_sample": int(raw_index),
                "source_start_seconds": float(raw_index / raw_sample_rate),
                "active_start_seconds": float(start_index / raw_sample_rate),
                "active_end_seconds": float(end_index / raw_sample_rate),
                "source_rms": window_rms(window),
                "noise_rms": float(quality["noise_rms"]),
                "tx_rms": float(quality["tx_rms"]),
                "snr_like_ratio": float(quality["tx_to_noise_rms_ratio"]),
                "estimated_snr_db": float(snr_info["estimated_snr_db"]),
                "spectral_peak_prominence": float(quality["spectral_peak_prominence"]),
                "clipping_rate": float(quality["clipping_rate"]),
                "quality_pass": bool(quality["quality_pass"]),
                "payload_seed": int(capture_meta.get("payload_seed", -1)),
                "random_payload_bits": bool(capture_meta.get("random_payload_bits", True)),
            }
            if len(manifest) < MANIFEST_PREVIEW_LIMIT:
                manifest.append(row)
            append_manifest_csv(
                manifest_csv,
                [
                    row["path"],
                    row["split"],
                    row["modulation"],
                    row["session_id"],
                    row["source"],
                    f"{row['source_start_seconds']:.8f}",
                    f"{row['active_start_seconds']:.8f}",
                    f"{row['active_end_seconds']:.8f}",
                    f"{row['source_rms']:.8f}",
                    f"{row['noise_rms']:.8f}",
                    f"{row['tx_rms']:.8f}",
                    f"{row['snr_like_ratio']:.8f}",
                    f"{row['spectral_peak_prominence']:.8f}",
                    f"{row['clipping_rate']:.8f}",
                    str(row["quality_pass"]),
                ],
            )
            counters[split] += 1
            produced += 1
        if produced == 0:
            print(f"warning=no_windows source={path}")
    balance_report = balance_dataset_files(out) if balance_after_import else {"enabled": False}
    final_counts = count_dataset_files(out)
    write_json(
        out / "manifest_oshea2018_ota.json",
        {
            "raw_written_counts_before_balance": counters,
            "counts": final_counts,
            "session_to_split": session_to_split,
            "windowing": {
                "window_len": window_len,
                "stride": stride,
                "examples_per_capture_limit": examples_per_capture_limit,
                "raw_sample_rate": raw_sample_rate,
                "target_sample_rate": target_sample_rate,
                "channel_bandwidth_hz": channel_bandwidth_hz,
                "window_start_seconds": window_start_seconds,
                "window_end_seconds": window_end_seconds,
                "active_start_seconds": cfg["ota"].get("active_start_seconds", None),
                "active_duration_seconds": cfg["ota"].get("active_duration_seconds", None),
                "window_sampling": sampling_mode,
                "active_region_detection": active_cfg,
                "quality_filter": quality_cfg,
                "balance_after_import": balance_after_import,
            },
            "rejected_captures": rejected,
            "balance_report": balance_report,
            "samples_manifest_csv": str(manifest_csv),
            "samples_preview": manifest,
            "samples_preview_limit": MANIFEST_PREVIEW_LIMIT,
        },
    )
    return out


def append_manifest_csv(path: Path, values: list[object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(str(value).replace(",", ";") for value in values) + "\n")


def count_dataset_files(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        counts[split] = {}
        for modulation in CLASS_NAMES:
            counts[split][modulation] = len(list(split_dir.glob(f"*_{modulation.lower()}_*.npz")))
    return counts


def balance_dataset_files(root: Path) -> dict[str, object]:
    report: dict[str, object] = {"enabled": True, "splits": {}}
    for split in ("train", "val", "test"):
        split_dir = root / split
        groups = {modulation: sorted(split_dir.glob(f"*_{modulation.lower()}_*.npz")) for modulation in CLASS_NAMES}
        counts = {modulation: len(files) for modulation, files in groups.items()}
        nonzero = [count for count in counts.values() if count > 0]
        if len(nonzero) != len(CLASS_NAMES):
            report["splits"][split] = {"counts_before": counts, "balanced": False, "reason": "one_or_more_classes_empty", "removed": 0}
            continue
        target = min(nonzero)
        removed = 0
        for modulation, files in groups.items():
            keep = set(uniform_keep(files, target))
            for path in files:
                if path not in keep:
                    path.unlink()
                    removed += 1
        counts_after = {modulation: len(list(split_dir.glob(f"*_{modulation.lower()}_*.npz"))) for modulation in CLASS_NAMES}
        report["splits"][split] = {"counts_before": counts, "counts_after": counts_after, "target_per_class": target, "balanced": True, "removed": removed}
    return report


def uniform_keep(files: list[Path], target: int) -> list[Path]:
    if len(files) <= target:
        return files
    positions = np.linspace(0, len(files) - 1, num=target)
    return [files[int(round(pos))] for pos in positions]


def detect_active_region(
    iq: np.ndarray,
    sample_rate: float,
    window_len: int,
    rx_lead_seconds: float,
    fallback_start: int,
    fallback_end: int,
    cfg: dict[str, object],
) -> dict[str, object]:
    if not bool(cfg.get("enabled", True)):
        return {
            "active_region_found": False,
            "active_start_sample": fallback_start,
            "active_end_sample": fallback_end,
            "active_detection_reason": "disabled",
        }
    block_samples = max(window_len, int(round(float(cfg.get("block_ms", 5.0)) * sample_rate / 1000.0)))
    hop_samples = max(window_len // 2, int(round(float(cfg.get("hop_ms", 2.5)) * sample_rate / 1000.0)))
    pre_end = min(len(iq), max(block_samples, int(round(rx_lead_seconds * sample_rate))))
    noise = iq[:pre_end]
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2))) if len(noise) else 0.0
    min_ratio = float(cfg.get("min_rms_ratio", 1.2))
    min_extra = float(cfg.get("min_absolute_rms_extra", 0.0))
    pad_seconds = float(cfg.get("pad_seconds", 0.05))
    starts: list[int] = []
    rms_values: list[float] = []
    for start in range(0, max(1, len(iq) - block_samples + 1), hop_samples):
        block = iq[start : start + block_samples]
        rms = float(np.sqrt(np.mean(np.abs(block) ** 2)))
        starts.append(start)
        rms_values.append(rms)
    if not starts:
        return {
            "active_region_found": False,
            "active_start_sample": fallback_start,
            "active_end_sample": fallback_end,
            "active_detection_reason": "empty_capture",
        }
    threshold = max(noise_rms * min_ratio, noise_rms + min_extra)
    active = np.asarray(rms_values) >= threshold
    ignore_until = int(round(rx_lead_seconds * sample_rate * float(cfg.get("ignore_lead_fraction", 0.8))))
    active = active & (np.asarray(starts) >= ignore_until)
    if not np.any(active):
        return {
            "active_region_found": False,
            "active_start_sample": fallback_start,
            "active_end_sample": fallback_end,
            "active_detection_reason": "no_rms_region_above_threshold",
            "active_threshold_rms": threshold,
            "active_noise_rms": noise_rms,
        }
    active_indices = np.where(active)[0]
    pad = int(round(pad_seconds * sample_rate))
    start_sample = max(fallback_start, int(starts[int(active_indices[0])] - pad))
    end_sample = min(fallback_end, int(starts[int(active_indices[-1])] + block_samples + pad))
    if end_sample - start_sample < window_len:
        start_sample, end_sample = fallback_start, fallback_end
    return {
        "active_region_found": True,
        "active_start_sample": int(start_sample),
        "active_end_sample": int(end_sample),
        "active_detection_reason": "rms_threshold",
        "active_threshold_rms": threshold,
        "active_noise_rms": noise_rms,
    }


def capture_quality(iq: np.ndarray, sample_rate: float, start_index: int, end_index: int, rx_lead_seconds: float, cfg: dict[str, object]) -> dict[str, object]:
    pre_end = min(len(iq), max(1, int(round(rx_lead_seconds * sample_rate))))
    noise = iq[:pre_end]
    active = iq[start_index:end_index]
    noise_rms = window_rms(noise)
    tx_rms = window_rms(active)
    ratio = float(tx_rms / max(noise_rms, 1e-12))
    peak_prominence = spectral_peak_prominence(active)
    clipping = clipping_rate(active)
    min_ratio = float(cfg.get("min_tx_to_noise_rms_ratio", 1.05))
    min_peak = float(cfg.get("min_spectral_peak_prominence", 0.0))
    max_clip = float(cfg.get("max_clipping_rate", 0.02))
    reasons: list[str] = []
    if ratio < min_ratio:
        reasons.append(f"low_tx_to_noise_rms_ratio:{ratio:.3f}<{min_ratio:.3f}")
    if peak_prominence < min_peak:
        reasons.append(f"low_spectral_peak_prominence:{peak_prominence:.3f}<{min_peak:.3f}")
    if clipping > max_clip:
        reasons.append(f"high_clipping_rate:{clipping:.3f}>{max_clip:.3f}")
    return {
        "noise_rms": noise_rms,
        "tx_rms": tx_rms,
        "tx_to_noise_rms_ratio": ratio,
        "spectral_peak_prominence": peak_prominence,
        "clipping_rate": clipping,
        "quality_pass": len(reasons) == 0,
        "quality_reason": "pass" if not reasons else ";".join(reasons),
    }


def capture_quality_against_noise(noise_iq: np.ndarray, active_iq: np.ndarray, sample_rate: float, cfg: dict[str, object]) -> dict[str, object]:
    noise_rms = window_rms(noise_iq)
    tx_rms = window_rms(active_iq)
    ratio = float(tx_rms / max(noise_rms, 1e-12))
    peak_prominence = spectral_peak_prominence(active_iq)
    clipping = clipping_rate(active_iq)
    min_ratio = float(cfg.get("min_tx_to_noise_rms_ratio", 2.0))
    min_peak = float(cfg.get("min_spectral_peak_prominence", 0.0))
    max_clip = float(cfg.get("max_clipping_rate", 0.02))
    min_snr_db = cfg.get("min_estimated_snr_db", None)
    snr = estimate_snr_db(noise_iq, active_iq)["estimated_snr_db"]
    reasons: list[str] = []
    if ratio < min_ratio:
        reasons.append(f"low_tx_to_noise_rms_ratio:{ratio:.3f}<{min_ratio:.3f}")
    if peak_prominence < min_peak:
        reasons.append(f"low_spectral_peak_prominence:{peak_prominence:.3f}<{min_peak:.3f}")
    if min_snr_db is not None and snr < float(min_snr_db):
        reasons.append(f"low_estimated_snr_db:{snr:.2f}<{float(min_snr_db):.2f}")
    if clipping > max_clip:
        reasons.append(f"high_clipping_rate:{clipping:.3f}>{max_clip:.3f}")
    return {
        "noise_rms": noise_rms,
        "tx_rms": tx_rms,
        "tx_to_noise_rms_ratio": ratio,
        "spectral_peak_prominence": peak_prominence,
        "clipping_rate": clipping,
        "quality_pass": len(reasons) == 0,
        "quality_reason": "pass" if not reasons else ";".join(reasons),
    }


def window_rms(iq: np.ndarray) -> float:
    if len(iq) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(iq) ** 2)))


def spectral_peak_prominence(iq: np.ndarray) -> float:
    if len(iq) < 16:
        return 0.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq.astype(np.complex64)))) ** 2
    power = np.log1p(spectrum).astype(np.float32)
    return float((np.max(power) - np.median(power)) / (np.std(power) + 1e-8))


def clipping_rate(iq: np.ndarray) -> float:
    if len(iq) == 0:
        return 0.0
    real = np.abs(np.real(iq))
    imag = np.abs(np.imag(iq))
    return float(np.mean((real >= 0.98) | (imag >= 0.98)))


def window_region(
    total_samples: int,
    sample_rate: float,
    window_len: int,
    start_seconds: float,
    end_seconds: float | None,
    guard_seconds: float,
) -> tuple[int, int]:
    start_index = max(0, int(round(start_seconds * sample_rate)))
    if end_seconds is None:
        end_index = max(0, total_samples - int(round(guard_seconds * sample_rate)))
    else:
        end_index = min(total_samples, int(round(end_seconds * sample_rate)))
    end_index = min(total_samples, max(start_index + window_len, end_index))
    return start_index, end_index


def window_indices(start_index: int, end_index: int, window_len: int, stride: int, limit: int, mode: str) -> list[int]:
    max_start = end_index - window_len
    if max_start < start_index:
        return []
    all_indices = list(range(start_index, max_start + 1, max(1, stride)))
    if limit <= 0 or len(all_indices) <= limit:
        return all_indices
    if mode in {"uniform", "uniform_tx_region", "even"}:
        positions = np.linspace(0, len(all_indices) - 1, num=limit)
        selected = sorted({all_indices[int(round(pos))] for pos in positions})
        if len(selected) < limit:
            for idx in all_indices:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= limit:
                    break
            selected.sort()
        return selected[:limit]
    if mode in {"first", "head"}:
        return all_indices[:limit]
    if mode in {"random", "random_tx_region"}:
        rng = np.random.default_rng(42)
        selected = rng.choice(np.asarray(all_indices), size=limit, replace=False)
        return sorted(int(idx) for idx in selected)
    raise ValueError(f"Unsupported window_sampling: {mode}")


def load_session_metadata(session_dir: Path) -> dict[str, object]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    keep = [
        "center_freq",
        "baseband_offset_hz",
        "distance_m",
        "agc",
        "tx_vga_gain",
        "tx_amp_gain",
        "rx_gain",
        "noise_only_file",
        "random_payload_bits",
        "payload_seed_policy",
    ]
    return {key: data[key] for key in keep if key in data}


def load_capture_metadata(session_dir: Path, capture_path: Path) -> dict[str, object]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data.get("captures", []):
        if Path(str(item.get("file", ""))).name == capture_path.name:
            result = dict(item)
            for key in ("baseband_offset_hz", "tx_vga_gain", "tx_amp_gain", "rx_gain"):
                if key in data and key not in result:
                    result[key] = data[key]
            return result
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    root = import_oshea2018_ota_windows(args.config, args.raw_root, args.output_root, clean=args.clean)
    print(f"OTA windows written to {root}")


if __name__ == "__main__":
    main()
