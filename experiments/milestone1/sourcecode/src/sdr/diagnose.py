from __future__ import annotations

import argparse

from src.sdr.soapy_common import require_soapy


def diagnose() -> None:
    SoapySDR, _, _, _ = require_soapy()
    devices = [dict(item) for item in SoapySDR.Device.enumerate()]
    print("SoapySDR devices:")
    if not devices:
        print("  none")
    for idx, dev in enumerate(devices):
        print(f"  [{idx}] {dev}")
    for driver in ("rtlsdr", "hackrf"):
        try:
            dev = SoapySDR.Device(f"driver={driver}")
            print(f"open driver={driver}: ok")
            del dev
        except Exception as exc:
            print(f"open driver={driver}: failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    diagnose()


if __name__ == "__main__":
    main()
