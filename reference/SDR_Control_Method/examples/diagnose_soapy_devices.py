from __future__ import annotations


def main() -> None:
    try:
        import SoapySDR  # type: ignore
    except Exception as exc:
        raise SystemExit(f"SoapySDR Python binding is unavailable: {exc}") from exc

    devices = [dict(item) for item in SoapySDR.Device.enumerate()]
    print("SoapySDR devices:")
    for idx, item in enumerate(devices):
        print(f"  [{idx}] {item}")

    for driver in ("rtlsdr", "hackrf"):
        try:
            dev = SoapySDR.Device(f"driver={driver}")
            del dev
            print(f"open driver={driver}: ok")
        except Exception as exc:
            print(f"open driver={driver}: failed: {exc}")


if __name__ == "__main__":
    main()
