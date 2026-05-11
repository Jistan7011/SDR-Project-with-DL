from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.run_exp02_capture_session import run_session
from src.common import write_json


def run_exp03_session(
    session_id: str,
    output_root: str = "../data/raw_iq",
    config_path: str = "../config/config.exp03.yaml",
    distance_m: float = 1.0,
    antenna_layout: str = "face_to_face",
    tx_antenna_orientation: str = "fixed",
    rx_antenna_orientation: str = "fixed",
    tx_height_cm: float | None = None,
    rx_height_cm: float | None = None,
    line_of_sight: bool = True,
    near_metal_objects: str = "minimized",
    human_nearby: str = "minimized",
    **kwargs: object,
) -> Path:
    dry_run = bool(kwargs.get("dry_run", False))
    session_dir = run_session(
        session_id=session_id,
        output_root=output_root,
        config_path=config_path,
        **kwargs,
    )
    if dry_run:
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        try:
            session_dir.rmdir()
        except OSError:
            pass
        print("Dry run complete; Experiment 3 metadata was not updated.")
        return session_dir
    metadata_path = session_dir / "metadata.json"
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "experiment": "exp3",
            "purpose": "1m_ota_distance_generalization_feature_fusion",
            "rf_path": "ota_antenna",
            "rf_cable_between_sdr": False,
            "tx_usb_connected_to_pc": True,
            "rx_usb_connected_to_pc": True,
            "tx_rx_distance_m": float(distance_m),
            "antenna_layout": antenna_layout,
            "tx_antenna_orientation": tx_antenna_orientation,
            "rx_antenna_orientation": rx_antenna_orientation,
            "tx_height_cm": tx_height_cm,
            "rx_height_cm": rx_height_cm,
            "line_of_sight": bool(line_of_sight),
            "near_metal_objects": near_metal_objects,
            "human_nearby": human_nearby,
        }
    )
    write_json(metadata_path, metadata)
    print(f"Experiment 3 OTA metadata updated at {metadata_path}")
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", default="../data/raw_iq")
    parser.add_argument("--config", default="../config/config.exp03.yaml")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--payloads", nargs="*", default=None)
    parser.add_argument("--baseband-offset-hz", type=float, default=500_000.0)
    parser.add_argument("--tx-vga-gain", type=float, default=None)
    parser.add_argument("--tx-amp-gain", type=float, default=None)
    parser.add_argument("--rx-gain", type=float, default=None)
    parser.add_argument("--tx-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--noise-seconds", type=float, default=3.0)
    parser.add_argument("--tx-start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--distance-m", type=float, default=1.0)
    parser.add_argument("--antenna-layout", default="face_to_face")
    parser.add_argument("--tx-antenna-orientation", default="fixed")
    parser.add_argument("--rx-antenna-orientation", default="fixed")
    parser.add_argument("--tx-height-cm", type=float, default=None)
    parser.add_argument("--rx-height-cm", type=float, default=None)
    parser.add_argument("--line-of-sight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--near-metal-objects", default="minimized")
    parser.add_argument("--human-nearby", default="minimized")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_exp03_session(
        session_id=args.session_id,
        output_root=args.output_root,
        config_path=args.config,
        distance_m=args.distance_m,
        antenna_layout=args.antenna_layout,
        tx_antenna_orientation=args.tx_antenna_orientation,
        rx_antenna_orientation=args.rx_antenna_orientation,
        tx_height_cm=args.tx_height_cm,
        rx_height_cm=args.rx_height_cm,
        line_of_sight=args.line_of_sight,
        near_metal_objects=args.near_metal_objects,
        human_nearby=args.human_nearby,
        radioconda_root=args.radioconda_root,
        payloads=args.payloads,
        baseband_offset_hz=args.baseband_offset_hz,
        tx_vga_gain=args.tx_vga_gain,
        tx_amp_gain=args.tx_amp_gain,
        rx_gain=args.rx_gain,
        tx_seconds=args.tx_seconds,
        capture_seconds=args.capture_seconds,
        noise_seconds=args.noise_seconds,
        tx_start_delay_seconds=args.tx_start_delay_seconds,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
