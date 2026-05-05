from __future__ import annotations

import argparse

import numpy as np
import torch

from src.common import CLASS_NAMES
from src.models.factory import build_model
from src.signal.demod import recover_payload
from src.signal.processing import complex64_from_channels


def decode(input_path: str, checkpoint_path: str) -> dict[str, object]:
    data = np.load(input_path, allow_pickle=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config") or {"model": {"type": "cnn1d", "input_channels": 2, "num_classes": len(CLASS_NAMES), "dropout": 0.3}}
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = torch.from_numpy(data["iq"].astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1).item())
        confidence = float(torch.softmax(logits, dim=1)[0, pred_idx].item())
    modulation = CLASS_NAMES[pred_idx]
    iq = data["raw_iq"].astype(np.complex64) if "raw_iq" in data.files else complex64_from_channels(data["iq"])
    sample_rate = float(data["sample_rate"])
    symbol_rate = float(data["symbol_rate"])
    sps = max(1, int(round(sample_rate / symbol_rate)))
    if "payload_bytes" in data.files:
        payload_bytes = int(data["payload_bytes"])
    elif "payload" in data.files:
        payload_bytes = len(str(data["payload"]))
    else:
        payload_bytes = 1
    recovered = recover_payload(modulation, iq, sps, sample_rate, payload_bytes=payload_bytes)
    fallback = None
    if not recovered.get("crc_ok") and "modulation" in data.files:
        expected_modulation = str(data["modulation"])
        if expected_modulation != modulation:
            fallback = recover_payload(expected_modulation, iq, sps, sample_rate, payload_bytes=payload_bytes)
            fallback = summarize_recovery(fallback) | {"modulation": expected_modulation}
    result = {
        "predicted_modulation": modulation,
        "confidence": confidence,
        "recovered": summarize_recovery(recovered),
        "metadata_assisted_recovery": fallback,
    }
    print(result)
    return result


def summarize_recovery(recovered: dict[str, object]) -> dict[str, object]:
    return {
        "payload": recovered.get("payload", ""),
        "crc_ok": bool(recovered.get("crc_ok", False)),
        "start": int(recovered.get("start", -1)),
        "bit_count": int(len(recovered.get("recovered_bits", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    decode(args.input, args.checkpoint)


if __name__ == "__main__":
    main()
