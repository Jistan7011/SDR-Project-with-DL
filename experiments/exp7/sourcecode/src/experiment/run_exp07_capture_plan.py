from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_config
from src.experiment.run_exp07_capture_session import run_exp07_capture_session


def run_exp07_capture_plan(
    config_path: str = "../config/config.exp07.yaml",
    output_root: str = "../data/raw_iq",
    radioconda_root: str = r"C:\Users\qus70\radioconda",
    first_session: int | None = None,
    last_session: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> None:
    cfg = load_config(config_path)
    exp_cfg = cfg.get("experiment7", {})
    plan = list(exp_cfg.get("session_plan", []))
    if first_session is None:
        first_session = int(exp_cfg.get("new_sessions", {}).get("first", 31))
    if last_session is None:
        last_session = int(exp_cfg.get("new_sessions", {}).get("last", 60))
    selected = []
    for item in plan:
        session_id = str(item["session_id"])
        number = int(session_id.split("_")[-1])
        if first_session <= number <= last_session:
            selected.append(item)
    if not selected:
        raise SystemExit(f"No exp7 sessions selected in range {first_session}..{last_session}")

    root = Path(output_root)
    for item in selected:
        session_id = str(item["session_id"])
        session_dir = root / session_id
        if skip_existing and (session_dir / "metadata.json").exists():
            print(f"SKIP existing {session_id}: {session_dir}")
            continue
        run_exp07_capture_session(
            session_id=session_id,
            output_root=output_root,
            config_path=config_path,
            radioconda_root=radioconda_root,
            tx_vga_gain=float(item.get("tx_vga_gain", 30.0)),
            rx_gain=float(item.get("rx_gain", 30.0)),
            baseband_offset_hz=float(item.get("baseband_offset_hz", 500_000.0)),
            antenna_note=str(item.get("antenna_note", "")),
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    parser.add_argument("--output-root", default="../data/raw_iq")
    parser.add_argument("--radioconda-root", default=r"C:\Users\qus70\radioconda")
    parser.add_argument("--first-session", type=int, default=None)
    parser.add_argument("--last-session", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    run_exp07_capture_plan(
        config_path=args.config,
        output_root=args.output_root,
        radioconda_root=args.radioconda_root,
        first_session=args.first_session,
        last_session=args.last_session,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
