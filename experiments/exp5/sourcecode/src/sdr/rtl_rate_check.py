from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_rtl_rate_check(sample_rate: float = 2.4e6, seconds: float = 10.0, radioconda_root: str | None = None) -> dict[str, object]:
    command = ["rtl_test", "-s", format_rate(sample_rate)]
    if radioconda_root:
        activate = str(Path(radioconda_root) / "Scripts" / "activate.bat")
        command = ["cmd", "/c", "call", activate, radioconda_root, "&&", *command]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=max(1.0, seconds))
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        return parse_rtl_output(output, sample_rate, timed_out=True)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    result = parse_rtl_output(output, sample_rate, timed_out=timed_out)
    result["returncode"] = proc.returncode
    return result


def parse_rtl_output(output: str, sample_rate: float, timed_out: bool = False) -> dict[str, object]:
    lowered = output.lower()
    drop_markers = ["lost at least", "dropped", "underrun", "overrun"]
    has_drop = any(marker in lowered for marker in drop_markers)
    has_device = "found" in lowered or "using device" in lowered
    return {
        "sample_rate": sample_rate,
        "timed_out_after_gate": timed_out,
        "device_seen": has_device,
        "drop_detected": has_drop,
        "recommended_rx_sample_rate": 2.048e6 if has_drop else sample_rate,
        "raw_output": output,
    }


def format_rate(sample_rate: float) -> str:
    return f"{sample_rate:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=float, default=2.4e6)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--radioconda-root", default=None)
    args = parser.parse_args()
    result = run_rtl_rate_check(args.sample_rate, args.seconds, args.radioconda_root)
    print(f"drop_detected={result['drop_detected']} recommended_rx_sample_rate={result['recommended_rx_sample_rate']}")
    if result["raw_output"]:
        print(result["raw_output"])


if __name__ == "__main__":
    main()
