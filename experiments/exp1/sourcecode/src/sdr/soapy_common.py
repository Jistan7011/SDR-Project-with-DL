from __future__ import annotations


def require_soapy():
    try:
        import SoapySDR  # type: ignore
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "SoapySDR Python bindings are not installed. Install SoapySDR and the HackRF/RTL-SDR drivers, "
            "then retry this command."
        ) from exc
    return SoapySDR, SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX
