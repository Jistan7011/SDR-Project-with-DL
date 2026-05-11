from __future__ import annotations

import argparse

import numpy as np
import torch

from src.common import CLASS_NAMES
from src.models.cnn1d import CNN1DClassifier
from src.signal.demod import recover_payload
from src.signal.processing import complex64_from_channels


def decode(input_path: str, checkpoint_path: str) -> dict[str, object]:
    data = np.load(input_path, allow_pickle=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = CNN1DClassifier(num_classes=len(CLASS_NAMES))
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
    payload_bytes = len(str(data["payload"])) if "payload" in data.files else 1
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
