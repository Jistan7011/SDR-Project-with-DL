from __future__ import annotations

import argparse

from src.common import load_config
from src.experiment.run_exp04_recovery_session import run_exp04_recovery_session


def run_exp04_recovery_plan(
    config_path: str = "../config/config.exp04.yaml",
    output_root: str = "../data/raw_iq",
    radioconda_root: str = r"C:\Users\qus70\radioconda",
    tx_seconds: float = 5.0,
    capture_seconds: float = 10.0,
    noise_seconds: float = 3.0,
    dry_run: bool = False,
) -> None:
    cfg = load_config(config_path)
    sessions = list(cfg.get("experiment4", {}).get("session_plan", []))
    if not sessions:
        raise SystemExit("No experiment4.session_plan entries found in config.")
    for item in sessions:
        run_exp04_recovery_session(
            session_id=str(item["session_id"]),
            output_root=output_root,
            config_path=config_path,
            radioconda_root=radioconda_root,
            distance_m=float(item.get("distance_m", cfg["experiment4"].get("tx_rx_distance_m", 1.0))),
            antenna_layout=str(item.get("antenna_layout", cfg["experiment4"].get("antenna_layout", "face_to_face"))),
            tx_vga_gain=float(item.get("tx_vga_gain", cfg["sdr"].get("tx_vga_gain", 30))),
            rx_gain=float(item.get("rx_gain", cfg["sdr"].get("rx_gain", 30))),
            baseband_offset_hz=float(item.get("baseband_offset_hz", cfg["sdr"].get("baseband_offset_hz", 500_000))),
            tx_seconds=tx_seconds,
            capture_seconds=capture_seconds,
            noise_seconds=noise_seconds,
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp04.yaml")
    parser.add_argument("--output-root", default="../data/raw_iq")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--tx-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--noise-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_exp04_recovery_plan(
        config_path=args.config,
        output_root=args.output_root,
        radioconda_root=args.radioconda_root,
        tx_seconds=args.tx_seconds,
        capture_seconds=args.capture_seconds,
        noise_seconds=args.noise_seconds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
