from __future__ import annotations

import argparse
from pathlib import Path

from src.common import write_json
from src.experiment.run_exp02_capture_session import DEFAULT_PAYLOAD_POOL, run_session


def run_exp04_recovery_session(
    session_id: str,
    output_root: str = "../data/raw_iq",
    config_path: str = "../config/config.exp04.yaml",
    radioconda_root: str = r"C:\Users\qus70\radioconda",
    distance_m: float = 1.0,
    antenna_layout: str = "face_to_face",
    tx_vga_gain: float = 30.0,
    rx_gain: float = 30.0,
    baseband_offset_hz: float = 500_000.0,
    tx_seconds: float = 5.0,
    capture_seconds: float = 10.0,
    noise_seconds: float = 3.0,
    dry_run: bool = False,
) -> Path:
    session_dir = run_session(
        session_id=session_id,
        output_root=output_root,
        config_path=config_path,
        radioconda_root=radioconda_root,
        payloads=DEFAULT_PAYLOAD_POOL,
        baseband_offset_hz=baseband_offset_hz,
        tx_vga_gain=tx_vga_gain,
        rx_gain=rx_gain,
        tx_seconds=tx_seconds,
        capture_seconds=capture_seconds,
        noise_seconds=noise_seconds,
        dry_run=dry_run,
    )
    if dry_run:
        return session_dir
    metadata_path = session_dir / "metadata.json"
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "exp4_end_to_end_recovery",
            "purpose": "1m OTA modulation classification plus payload recovery",
            "rf_path": "ota_antenna",
            "rf_cable_between_sdr": False,
            "tx_usb_connected_to_pc": True,
            "rx_usb_connected_to_pc": True,
            "tx_rx_distance_m": float(distance_m),
            "antenna_layout": antenna_layout,
            "line_of_sight": True,
            "payload_recovery_frame": "preamble16_sync8_payload8_crc8",
        }
    )
    write_json(metadata_path, metadata)
    print(f"Experiment 4 metadata updated: {metadata_path}")
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", default="../data/raw_iq")
    parser.add_argument("--config", default="../config/config.exp04.yaml")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--distance-m", type=float, default=1.0)
    parser.add_argument("--antenna-layout", default="face_to_face")
    parser.add_argument("--tx-vga-gain", type=float, default=30.0)
    parser.add_argument("--rx-gain", type=float, default=30.0)
    parser.add_argument("--baseband-offset-hz", type=float, default=500_000.0)
    parser.add_argument("--tx-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--noise-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_exp04_recovery_session(
        session_id=args.session_id,
        output_root=args.output_root,
        config_path=args.config,
        radioconda_root=args.radioconda_root,
        distance_m=args.distance_m,
        antenna_layout=args.antenna_layout,
        tx_vga_gain=args.tx_vga_gain,
        rx_gain=args.rx_gain,
        baseband_offset_hz=args.baseband_offset_hz,
        tx_seconds=args.tx_seconds,
        capture_seconds=args.capture_seconds,
        noise_seconds=args.noise_seconds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
