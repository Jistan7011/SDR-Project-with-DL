from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.models.cnn1d import build_model
from src.signal.demod import recover_payload
from src.signal.frame import text_to_bits
from src.signal.processing import complex64_from_channels, instantaneous_frequency_channel, iq_to_channels, magnitude_channel, psd_feature
from src.signal.recovery_metrics import bit_error_rate, character_error_rate, failure_stage, packet_success


def evaluate_recovery(
    input_root: str,
    checkpoint: str,
    output_dir: str,
    config_path: str = "../config/config.exp04.yaml",
    splits: list[str] | None = None,
    max_samples: int | None = None,
) -> dict[str, object]:
    cfg = load_config(config_path)
    model, feature_mode, window_size = load_classifier(checkpoint, cfg)
    files = collect_npz_files(Path(input_root), splits)
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise SystemExit(f"No .npz samples found under {input_root}")

    rows = [evaluate_one(path, model, feature_mode, window_size, cfg) for path in files]
    summary = summarize(rows)
    out = ensure_dir(output_dir)
    write_json(out / "recovery_eval.json", {"summary": summary, "samples": rows})
    write_csv(out / "recovery_eval.csv", rows)
    write_summary_md(out / "recovery_summary.md", summary)
    print(
        "classification_accuracy={classification_accuracy:.4f} "
        "payload_recovery_accuracy={payload_recovery_accuracy:.4f} "
        "oracle_crc_pass_rate={oracle_crc_pass_rate:.4f}".format(**summary)
    )
    return {"summary": summary, "samples": rows}


def load_classifier(checkpoint: str, cfg: dict[str, object]) -> tuple[torch.nn.Module, str, int]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    dataset_cfg = ckpt_cfg.get("dataset", cfg["dataset"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=int(model_cfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, str(dataset_cfg.get("feature_mode", cfg["dataset"].get("feature_mode", "iq"))), int(dataset_cfg.get("window_size", cfg["dataset"]["window_size"]))


def collect_npz_files(root: Path, splits: list[str] | None) -> list[Path]:
    if splits:
        files: list[Path] = []
        for split in splits:
            files.extend(sorted((root / split).glob("*.npz")))
        return files
    split_dirs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if split_dirs:
        files = []
        for split in sorted(split_dirs):
            files.extend(sorted(split.glob("*.npz")))
        return files
    return sorted(root.glob("*.npz"))


def evaluate_one(path: Path, model: torch.nn.Module, feature_mode: str, window_size: int, cfg: dict[str, object]) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    channels = data["iq"].astype(np.float32)
    x = make_feature_tensor(channels, feature_mode, window_size)
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None, :, :]))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)
    confidence = float(sorted_probs[-1])
    margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else confidence
    predicted_modulation = CLASS_NAMES[pred_idx]
    expected_modulation = str(data["modulation"])
    expected_payload = str(data["payload"])
    expected_payload_bits = text_to_bits(expected_payload)
    dsp_iq = data["raw_iq"].astype(np.complex64) if "raw_iq" in data.files else complex64_from_channels(channels)
    sample_rate = float(data["sample_rate"]) if "sample_rate" in data.files else float(cfg["dataset"]["target_sample_rate"])
    symbol_rate = float(data["symbol_rate"]) if "symbol_rate" in data.files else float(cfg["sdr"]["symbol_rate"])
    samples_per_symbol = max(1, int(round(sample_rate / symbol_rate)))
    carrier_freq = float(data["demod_carrier_hz"]) if "demod_carrier_hz" in data.files else 0.0

    recovered = recover_payload(predicted_modulation, dsp_iq, samples_per_symbol, sample_rate, carrier_freq=carrier_freq)
    oracle = recover_payload(expected_modulation, dsp_iq, samples_per_symbol, sample_rate, carrier_freq=carrier_freq)
    recovered_bits = np.asarray(recovered.get("bits", []), dtype=np.uint8)
    oracle_bits = np.asarray(oracle.get("bits", []), dtype=np.uint8)
    pkt_ok = packet_success(expected_payload, str(recovered.get("payload", "")), bool(recovered.get("crc_ok", False)))
    oracle_pkt_ok = packet_success(expected_payload, str(oracle.get("payload", "")), bool(oracle.get("crc_ok", False)))
    stage = failure_stage(
        expected_modulation,
        predicted_modulation,
        str(recovered.get("payload", "")),
        bool(recovered.get("crc_ok", False)),
        int(recovered.get("start", -1)),
        expected_payload,
    )
    return {
        "path": str(path),
        "session_id": scalar(data, "session_id", ""),
        "expected_modulation": expected_modulation,
        "predicted_modulation": predicted_modulation,
        "classifier_confidence": confidence,
        "classification_margin": margin,
        "expected_payload": expected_payload,
        "recovered_payload": str(recovered.get("payload", "")),
        "crc_ok": bool(recovered.get("crc_ok", False)),
        "packet_success": pkt_ok,
        "oracle_recovered_payload": str(oracle.get("payload", "")),
        "oracle_demod_crc_ok": bool(oracle.get("crc_ok", False)),
        "oracle_packet_success": oracle_pkt_ok,
        "failure_stage": stage,
        "ber": bit_error_rate(expected_payload_bits, recovered_bits),
        "oracle_ber": bit_error_rate(expected_payload_bits, oracle_bits),
        "cer": character_error_rate(expected_payload, str(recovered.get("payload", ""))),
        "sample_rate": sample_rate,
        "symbol_rate": symbol_rate,
        "baseband_offset_hz": scalar(data, "baseband_offset_hz", None),
        "estimated_snr_db": scalar(data, "estimated_snr_db", scalar(data, "snr_db", None)),
        "tx_vga_gain": scalar(data, "tx_vga_gain", None),
        "rx_gain": scalar(data, "rx_gain", None),
        "tx_rx_distance_m": scalar(data, "tx_rx_distance_m", None),
        "rf_path": scalar(data, "rf_path", "ota_antenna"),
        "rf_cable_between_sdr": scalar(data, "rf_cable_between_sdr", False),
    }


def make_feature_tensor(channels: np.ndarray, feature_mode: str, window_size: int) -> np.ndarray:
    if channels.shape[1] != window_size:
        channels = iq_to_channels(complex64_from_channels(channels), window_size)
    key = feature_mode.lower().replace("-", "_")
    base = channels[:2].astype(np.float32)
    complex_iq = complex64_from_channels(base)
    if key in {"iq", "i_q"}:
        return base
    if key in {"iq_ifreq", "iq_instfreq", "iq_instantaneous_frequency"}:
        return np.concatenate([base, instantaneous_frequency_channel(complex_iq)[None, :]], axis=0).astype(np.float32)
    if key in {"iq_mag", "iq_magnitude"}:
        return np.concatenate([base, magnitude_channel(complex_iq)[None, :]], axis=0).astype(np.float32)
    if key in {"iq_mag_ifreq", "iq_magnitude_instantaneous_frequency"}:
        return np.concatenate([base, magnitude_channel(complex_iq)[None, :], instantaneous_frequency_channel(complex_iq)[None, :]], axis=0).astype(np.float32)
    if key in {"iq_mag_ifreq_psd", "fusion"}:
        return np.concatenate([base, magnitude_channel(complex_iq)[None, :], instantaneous_frequency_channel(complex_iq)[None, :], psd_feature(complex_iq, bins=window_size)[None, :]], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def scalar(data: np.lib.npyio.NpzFile, key: str, default: object) -> object:
    if key not in data.files:
        return default
    value = data[key]
    return value.tolist() if hasattr(value, "tolist") else value


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_count": len(rows),
        "classification_accuracy": mean([row["expected_modulation"] == row["predicted_modulation"] for row in rows]),
        "crc_pass_rate": mean([bool(row["crc_ok"]) for row in rows]),
        "packet_success_rate": mean([bool(row["packet_success"]) for row in rows]),
        "payload_recovery_accuracy": mean([bool(row["packet_success"]) for row in rows]),
        "oracle_crc_pass_rate": mean([bool(row["oracle_demod_crc_ok"]) for row in rows]),
        "oracle_packet_success_rate": mean([bool(row["oracle_packet_success"]) for row in rows]),
        "mean_ber": mean([float(row["ber"]) for row in rows]),
        "mean_cer": mean([float(row["cer"]) for row in rows]),
        "mean_confidence": mean([float(row["classifier_confidence"]) for row in rows]),
        "failure_stage_counts": counts([str(row["failure_stage"]) for row in rows]),
    }


def mean(values: list[object]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Experiment 4 Recovery Summary",
        "",
        f"- sample_count: {summary['sample_count']}",
        f"- classification_accuracy: {summary['classification_accuracy']:.4f}",
        f"- payload_recovery_accuracy: {summary['payload_recovery_accuracy']:.4f}",
        f"- crc_pass_rate: {summary['crc_pass_rate']:.4f}",
        f"- oracle_crc_pass_rate: {summary['oracle_crc_pass_rate']:.4f}",
        f"- mean_ber: {summary['mean_ber']:.4f}",
        f"- mean_cer: {summary['mean_cer']:.4f}",
        f"- mean_confidence: {summary['mean_confidence']:.4f}",
        "",
        "## Failure Stages",
    ]
    for key, value in dict(summary["failure_stage_counts"]).items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="../config/config.exp04.yaml")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate_recovery(args.input_root, args.checkpoint, args.output_dir, args.config, args.splits, args.max_samples)


if __name__ == "__main__":
    main()
