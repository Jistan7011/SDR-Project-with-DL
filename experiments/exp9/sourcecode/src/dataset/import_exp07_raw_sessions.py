from __future__ import annotations

import argparse

from src.common import load_config
from src.dataset.import_exp05_classification_windows import import_exp05_classification_windows


def import_exp07_raw_sessions(
    raw_roots: list[str] | None = None,
    output_root: str = "../data/processed_retrained",
    config_path: str = "../config/config.exp07.yaml",
) -> None:
    cfg = load_config(config_path)
    roots = raw_roots or [str(path) for path in cfg.get("experiment7", {}).get("raw_roots", [])]
    if not roots:
        roots = ["../../exp4/data/raw_iq", "../../exp5/data/raw_iq", "../data/raw_iq"]
    exp_cfg = cfg.get("experiment7", {})
    import_exp05_classification_windows(
        raw_roots=roots,
        output_root=output_root,
        config_path=config_path,
        active_start_seconds=float(exp_cfg.get("active_start_seconds", 1.1)),
        active_duration_seconds=float(exp_cfg.get("active_duration_seconds", 5.0)),
        windows_per_capture=int(exp_cfg.get("classification_windows_per_capture", 20)),
        target_sample_rate=float(cfg["dataset"].get("target_sample_rate", 160000)),
        channel_bandwidth_hz=float(cfg["dataset"].get("channel_bandwidth_hz", 100000)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-roots", nargs="*", default=None)
    parser.add_argument("--output-root", default="../data/processed_retrained")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    args = parser.parse_args()
    import_exp07_raw_sessions(args.raw_roots, args.output_root, args.config)


if __name__ == "__main__":
    main()
