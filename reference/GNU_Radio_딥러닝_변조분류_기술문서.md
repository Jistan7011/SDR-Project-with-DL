# GNU Radio 기술문서 — 딥러닝 기반 변조 분류 및 데이터 복호화 시스템

> **실험 목표**  
> HackRF One(TX) + RTL-SDR V4(RX) 하드웨어를 GNU Radio로 제어하여  
> BASK / BFSK / BPSK 신호를 송신하고, 수신된 IQ 데이터로  
> 딥러닝 모델이 변조 방식을 분류하며 원본 비트열을 복호화한다.  
> AWGN 노이즈 추가 및 다중 신호 동시 송신 환경에서도 분류 가능하도록 성능을 업그레이드한다.

---

## 목차

1. [시스템 구성 개요](#1-시스템-구성-개요)
2. [하드웨어 설정 — HackRF TX / RTL-SDR V4 RX](#2-하드웨어-설정)
3. [이진 문자열 인코딩 — 비트열 생성](#3-이진-문자열-인코딩)
4. [변조 구현 — BASK / BFSK / BPSK](#4-변조-구현)
5. [AWGN 채널 모델 추가](#5-awgn-채널-모델-추가)
6. [다중 신호 동시 송신](#6-다중-신호-동시-송신)
7. [IQ 데이터 수신 및 캡처](#7-iq-데이터-수신-및-캡처)
8. [딥러닝 데이터셋 생성 파이프라인](#8-딥러닝-데이터셋-생성-파이프라인)
9. [딥러닝 모델 연동 — 실시간 분류](#9-딥러닝-모델-연동--실시간-분류)
10. [데이터 복호화 — 비트열 → 문자열](#10-데이터-복호화)
11. [성능 측정 — 분류 정확도 & SNR / 전송 성공률](#11-성능-측정)
12. [전체 플로우그래프 코드](#12-전체-플로우그래프-코드)
13. [GRC 블록 구성 참고](#13-grc-블록-구성-참고)

---

## 1. 시스템 구성 개요

```
[PC: GNU Radio TX]
  └─ 문자열 인코딩 (BASK1, BFSK1, BPSK1 ...)
  └─ 변조 블록 (BASK / BFSK / BPSK)
  └─ AWGN 채널 추가 (선택)
  └─ 다중 신호 합성 (선택)
  └─ HackRF One → 공중 RF 송신 (또는 케이블 직결)

[PC: GNU Radio RX]
  └─ RTL-SDR V4 → IQ 샘플 수신
  └─ IQ 파일 저장 (.bin)
  └─ 딥러닝 추론 블록 (Embedded Python Block)
       ├─ 변조 분류 (BASK / BFSK / BPSK)
       └─ 비트열 복호화 → 문자열 복원

[성능 통계]
  └─ 분류 정확도 (Accuracy)
  └─ SNR 측정
  └─ 비트 오류율 (BER) / 전송 성공률
```

### 주요 파라미터

| 항목 | 값 |
|------|-----|
| 중심 주파수 | 433 MHz (ISM 대역 권장) |
| 샘플레이트 (TX) | 2 Msps |
| 샘플레이트 (RX) | 2.048 Msps |
| 심볼 레이트 | 10 ksps |
| 비트당 샘플 수 | 200 |
| BASK 반송파 | 433.1 MHz |
| BFSK f0 (0비트) | 433.0 MHz |
| BFSK f1 (1비트) | 433.2 MHz |
| BPSK 반송파 | 433.1 MHz |

---

## 2. 하드웨어 설정

### 2.1 HackRF One 설정 (TX)

HackRF는 SoapySDR 드라이버를 통해 GNU Radio에서 제어한다.

```python
# ─── HackRF TX 소스 블록 ───────────────────────────────────────
from gnuradio import gr
from gnuradio.soapy import sink as soapy_sink

class HackRF_TX(gr.top_block):
    def __init__(self, center_freq=433.1e6, samp_rate=2e6, tx_gain=30):
        super().__init__()

        # SoapySDR HackRF 싱크
        self.hackrf_sink = soapy_sink(
            "",                  # 첫 번째 HackRF
            "hackrf",            # 드라이버 이름
            1,                   # 채널 수
            "",                  # FPGA 이미지 경로 (사용 안 함)
            "",
            [""],
            [""]
        )
        self.hackrf_sink.set_sample_rate(0, samp_rate)
        self.hackrf_sink.set_frequency(0, center_freq)
        self.hackrf_sink.set_gain(0, "AMP", 0)      # 0 또는 14 dB (RF Amp)
        self.hackrf_sink.set_gain(0, "VGA", tx_gain) # 0~47 dB
```

**HackRF 설치 확인:**
```bash
# SoapySDR HackRF 드라이버 확인
SoapySDRUtil --probe="driver=hackrf"

# hackrf_info 로 연결 확인
hackrf_info
```

### 2.2 RTL-SDR V4 설정 (RX)

```python
# ─── RTL-SDR V4 소스 블록 ─────────────────────────────────────
from gnuradio.soapy import source as soapy_source

class RTLSDR_RX(gr.top_block):
    def __init__(self, center_freq=433.1e6, samp_rate=2.048e6, gain=40):
        super().__init__()

        self.rtlsdr_source = soapy_source(
            "",
            "rtlsdr",
            1, "", "", [""], [""]
        )
        self.rtlsdr_source.set_sample_rate(0, samp_rate)
        self.rtlsdr_source.set_frequency(0, center_freq)
        self.rtlsdr_source.set_frequency_correction(0, 0)  # PPM 보정
        self.rtlsdr_source.set_gain_mode(0, False)         # 수동 이득
        self.rtlsdr_source.set_gain(0, "TUNER", gain)      # 0~49.6 dB
        self.rtlsdr_source.set_dc_offset_mode(0, True)     # DC 오프셋 제거
        self.rtlsdr_source.set_iq_balance_mode(0, True)    # IQ 불균형 보정
```

**RTL-SDR V4 전용 드라이버 설치:**
```bash
# RTL-SDR V4 는 rtl_biast 내장 칩셋을 사용 → 최신 드라이버 필요
sudo apt install rtl-sdr librtlsdr-dev

# V4 전용 드라이버 (공식 권장)
git clone https://github.com/rtlsdrblog/rtl-sdr-blog
cd rtl-sdr-blog && mkdir build && cd build
cmake .. -DINSTALL_UDEV_RULES=ON
make && sudo make install

# 권한 설정
sudo cp ../cmake/Modules/rtl-sdr.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 3. 이진 문자열 인코딩

송신 데이터는 `BASK1`, `BASK2`, `BFSK1`, `BPSK1` 등 변조 방식과 신호 번호를 포함한 문자열이다.  
각 문자를 ASCII → 8비트 이진수로 변환하여 심볼 스트림을 생성한다.

```python
import numpy as np

def string_to_bits(text: str) -> np.ndarray:
    """
    문자열을 비트 배열로 변환
    예: "BASK1" → [0,1,0,0,0,0,1,0, 0,1,0,0,0,0,0,1, ...]
    """
    bits = []
    for ch in text:
        byte = ord(ch)
        for i in range(7, -1, -1):       # MSB first
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=np.float32)

def bits_to_string(bits: np.ndarray) -> str:
    """비트 배열 → 문자열 복원"""
    bits = bits.astype(int)
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | int(b)
        chars.append(chr(byte))
    return ''.join(chars)

# 테스트
labels = ["BASK1", "BASK2", "BFSK1", "BFSK2", "BPSK1", "BPSK2"]
for lbl in labels:
    bits = string_to_bits(lbl)
    recovered = bits_to_string(bits)
    print(f"{lbl:6s} → {len(bits)} bits → 복원: {recovered}")
```

### 패킷 프레이밍 (프리앰블 + 데이터)

실제 수신 시 동기화를 위해 프리앰블(알려진 비트열)을 앞에 붙인다.

```python
PREAMBLE = np.array([1,0,1,0,1,0,1,0, 1,1,1,1,0,0,0,0], dtype=np.float32)

def make_packet(text: str) -> np.ndarray:
    """프리앰블 + 데이터 비트열 생성"""
    data_bits = string_to_bits(text)
    return np.concatenate([PREAMBLE, data_bits])

def find_preamble(bits: np.ndarray, threshold=14) -> int:
    """
    수신 비트열에서 프리앰블 위치 탐색
    threshold: 허용 오류 비트 수
    """
    plen = len(PREAMBLE)
    for i in range(len(bits) - plen):
        match = np.sum(bits[i:i+plen] == PREAMBLE)
        if match >= threshold:
            return i + plen    # 데이터 시작 인덱스
    return -1
```

---

## 4. 변조 구현

### 4.1 BASK (Binary Amplitude Shift Keying)

| 비트 | 신호 |
|------|------|
| `1` | A·cos(2πf·t) — 반송파 있음 |
| `0` | 0 (무신호) |

```python
import numpy as np
from gnuradio import gr

class BASK_Modulator(gr.sync_block):
    """
    BASK 변조기
    입력: float32 비트 스트림 (0 또는 1)
    출력: complex64 IQ 샘플
    """
    def __init__(self, samp_per_sym=200, carrier_freq=100e3, samp_rate=2e6):
        gr.sync_block.__init__(self,
            name="BASK Modulator",
            in_sig=[np.float32],
            out_sig=[np.complex64]
        )
        self.sps      = samp_per_sym
        self.fc       = carrier_freq
        self.fs       = samp_rate
        self.phase    = 0.0
        self.set_output_multiple(samp_per_sym)  # 출력 단위를 sps 배수로 고정

    def work(self, input_items, output_items):
        bits = input_items[0]
        out  = output_items[0]
        n_syms = min(len(bits), len(out) // self.sps)
        for i in range(n_syms):
            bit = bits[i]
            for k in range(self.sps):
                t = (i * self.sps + k) / self.fs
                amp = 1.0 if bit > 0.5 else 0.0
                out[i * self.sps + k] = amp * np.exp(1j * 2 * np.pi * self.fc * t)
        return n_syms * self.sps


# ─── NumPy 기반 오프라인 BASK 신호 생성 ───────────────────────
def generate_bask(bits, sps=200, fc=100e3, fs=2e6) -> np.ndarray:
    N = len(bits) * sps
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    envelope = np.repeat(bits.astype(np.float32), sps)
    return (envelope * carrier).astype(np.complex64)
```

### 4.2 BFSK (Binary Frequency Shift Keying)

| 비트 | 주파수 |
|------|--------|
| `0` | f0 = fc − Δf |
| `1` | f1 = fc + Δf |

```python
class BFSK_Modulator(gr.sync_block):
    """
    BFSK 변조기
    입력: float32 비트 스트림
    출력: complex64 IQ 샘플
    """
    def __init__(self, samp_per_sym=200, f0=80e3, f1=120e3, samp_rate=2e6):
        gr.sync_block.__init__(self,
            name="BFSK Modulator",
            in_sig=[np.float32],
            out_sig=[np.complex64]
        )
        self.sps   = samp_per_sym
        self.f0    = f0
        self.f1    = f1
        self.fs    = samp_rate
        self.set_output_multiple(samp_per_sym)

    def work(self, input_items, output_items):
        bits = input_items[0]
        out  = output_items[0]
        n_syms = min(len(bits), len(out) // self.sps)
        for i in range(n_syms):
            freq = self.f1 if bits[i] > 0.5 else self.f0
            for k in range(self.sps):
                t = (i * self.sps + k) / self.fs
                out[i * self.sps + k] = np.exp(1j * 2 * np.pi * freq * t)
        return n_syms * self.sps


# ─── NumPy 기반 오프라인 BFSK 생성 ────────────────────────────
def generate_bfsk(bits, sps=200, f0=80e3, f1=120e3, fs=2e6) -> np.ndarray:
    N = len(bits) * sps
    t = np.arange(N) / fs
    freq_seq = np.repeat(np.where(bits > 0.5, f1, f0), sps)
    # 연속 위상 FSK (CPFSK) — 위상 누적
    phase = np.cumsum(2 * np.pi * freq_seq / fs)
    return np.exp(1j * phase).astype(np.complex64)
```

### 4.3 BPSK (Binary Phase Shift Keying)

| 비트 | 위상 |
|------|------|
| `0` | 0° |
| `1` | 180° |

```python
class BPSK_Modulator(gr.sync_block):
    """
    BPSK 변조기
    입력: float32 비트 스트림
    출력: complex64 IQ 샘플
    """
    def __init__(self, samp_per_sym=200, carrier_freq=100e3, samp_rate=2e6):
        gr.sync_block.__init__(self,
            name="BPSK Modulator",
            in_sig=[np.float32],
            out_sig=[np.complex64]
        )
        self.sps = samp_per_sym
        self.fc  = carrier_freq
        self.fs  = samp_rate
        self.set_output_multiple(samp_per_sym)

    def work(self, input_items, output_items):
        bits = input_items[0]
        out  = output_items[0]
        n_syms = min(len(bits), len(out) // self.sps)
        for i in range(n_syms):
            phase = np.pi if bits[i] > 0.5 else 0.0
            for k in range(self.sps):
                t = (i * self.sps + k) / self.fs
                out[i * self.sps + k] = np.exp(1j * (2 * np.pi * self.fc * t + phase))
        return n_syms * self.sps


# ─── NumPy 기반 오프라인 BPSK 생성 ────────────────────────────
def generate_bpsk(bits, sps=200, fc=100e3, fs=2e6) -> np.ndarray:
    N = len(bits) * sps
    t = np.arange(N) / fs
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    phase_map = np.where(bits > 0.5, np.pi, 0.0)
    phases = np.repeat(phase_map, sps)
    return (carrier * np.exp(1j * phases)).astype(np.complex64)
```

### 4.4 GNU Radio 내장 블록 활용 (GRC 권장)

GRC에서 변조를 구현할 때는 내장 블록을 조합하는 것이 실용적이다.

```
[BASK GRC 구성]
Vector Source (bits) → Repeat (sps) → Multiply (× carrier) → HackRF Sink

[BFSK GRC 구성]
Vector Source (bits) → Chunks to Symbols (0→f0, 1→f1) → VCO → HackRF Sink

[BPSK GRC 구성]
Vector Source (bits) → Chunks to Symbols (0→+1, 1→-1) → BPSK Mod → HackRF Sink
```

---

## 5. AWGN 채널 모델 추가

GNU Radio의 내장 `channel_model` 블록 또는 직접 구현으로 AWGN을 추가한다.

### 5.1 GNU Radio channel_model 블록 사용

```python
from gnuradio import channels

# noise_voltage = sqrt(signal_power / (2 * SNR_linear))
def snr_to_noise_voltage(snr_db: float, signal_power: float = 1.0) -> float:
    snr_linear = 10 ** (snr_db / 10)
    return np.sqrt(signal_power / (2 * snr_linear))

class ChannelSim(gr.top_block):
    def __init__(self, snr_db=10.0):
        super().__init__()
        noise_v = snr_to_noise_voltage(snr_db)

        self.channel = channels.channel_model(
            noise_voltage=noise_v,    # AWGN 노이즈 전압
            frequency_offset=0.0,     # 주파수 오프셋 (정규화: Hz/samp_rate)
            epsilon=1.0,              # 타이밍 오프셋 (샘플레이트 비율)
            taps=[1.0 + 0j],          # 채널 임펄스 응답 (flat fading)
            noise_seed=42
        )
```

### 5.2 NumPy 기반 AWGN 직접 추가 (오프라인 데이터셋 생성용)

```python
def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    복소 신호에 AWGN 추가
    SNR(dB) = 10*log10(P_signal / P_noise)
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    # 복소 AWGN: 실수/허수 각각 noise_power/2
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return (signal + noise).astype(np.complex64)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """실제 SNR 측정 (dB)"""
    noise = noisy - clean
    p_sig   = np.mean(np.abs(clean) ** 2)
    p_noise = np.mean(np.abs(noise) ** 2)
    return 10 * np.log10(p_sig / (p_noise + 1e-12))


# ─── 다양한 SNR 수준에서 데이터셋 생성 ─────────────────────────
SNR_RANGE = [-5, 0, 5, 10, 15, 20]   # dB

def augment_with_snr(signal: np.ndarray, snr_list=SNR_RANGE):
    """하나의 신호를 여러 SNR 수준으로 복제 반환"""
    return {snr: add_awgn(signal, snr) for snr in snr_list}
```

---

## 6. 다중 신호 동시 송신

서로 다른 주파수 채널에 변조 신호를 배치하여 하나의 IQ 스트림으로 합성한다.

```python
# 각 신호에 서브캐리어 주파수 오프셋 지정
SIGNAL_CHANNELS = {
    "BASK1": {"mod": "BASK", "freq_offset": -300e3},
    "BASK2": {"mod": "BASK", "freq_offset": -100e3},
    "BFSK1": {"mod": "BFSK", "freq_offset": +100e3},
    "BPSK1": {"mod": "BPSK", "freq_offset": +300e3},
}

def upconvert(signal: np.ndarray, freq_offset: float, fs: float) -> np.ndarray:
    """신호를 freq_offset 만큼 주파수 이동"""
    t = np.arange(len(signal)) / fs
    return (signal * np.exp(1j * 2 * np.pi * freq_offset * t)).astype(np.complex64)

def make_composite_signal(fs=2e6, sps=200) -> np.ndarray:
    """
    여러 변조 신호를 다른 주파수에 배치하고 합산
    """
    channel_signals = []
    for label, cfg in SIGNAL_CHANNELS.items():
        bits = make_packet(label)
        mod  = cfg["mod"]
        if mod == "BASK":
            sig = generate_bask(bits, sps=sps, fc=0, fs=fs)   # 베이스밴드
        elif mod == "BFSK":
            sig = generate_bfsk(bits, sps=sps, f0=-20e3, f1=20e3, fs=fs)
        else:  # BPSK
            sig = generate_bpsk(bits, sps=sps, fc=0, fs=fs)
        sig_up = upconvert(sig, cfg["freq_offset"], fs)
        channel_signals.append(sig_up)

    # 길이 맞추기 (가장 짧은 길이에 맞춤)
    min_len = min(len(s) for s in channel_signals)
    composite = sum(s[:min_len] for s in channel_signals)
    # 정규화 (최대 진폭 0.9)
    composite = composite / (np.max(np.abs(composite)) + 1e-9) * 0.9
    return composite.astype(np.complex64)
```

### 6.1 GRC 다중 신호 합성 블록 구성

```
BASK 변조기 ──┐
BFSK 변조기 ──┤→ Add (blocks.add_cc) → Multiply Const (게인 조절) → HackRF Sink
BPSK 변조기 ──┘
```

GNU Radio 내장 `blocks.add_cc` 블록으로 여러 변조 신호를 합산한다.

```python
from gnuradio import blocks

# GRC Python 코드 예시
self.adder = blocks.add_cc(1)

self.connect(bask_mod, (self.adder, 0))
self.connect(bfsk_mod, (self.adder, 1))
self.connect(bpsk_mod, (self.adder, 2))
self.connect(self.adder, self.hackrf_sink)
```

---

## 7. IQ 데이터 수신 및 캡처

### 7.1 IQ 파일 저장 (실시간 캡처)

```python
from gnuradio import gr, blocks
from gnuradio.soapy import source as soapy_source

class IQ_Capture(gr.top_block):
    def __init__(self, filename="rx_capture.bin",
                 center_freq=433.1e6, samp_rate=2.048e6,
                 duration_sec=5.0):
        super().__init__()

        num_samples = int(samp_rate * duration_sec)

        # RTL-SDR 소스
        self.sdr = soapy_source("", "rtlsdr", 1, "", "", [""], [""])
        self.sdr.set_sample_rate(0, samp_rate)
        self.sdr.set_frequency(0, center_freq)
        self.sdr.set_gain(0, "TUNER", 40)

        # DC 오프셋 제거 (RTL-SDR 특성상 중심 주파수에 DC 스파이크 발생)
        self.dc_block = blocks.dc_blocker_cc(32, True)

        # 지정 샘플 수만 캡처
        self.head    = blocks.head(gr.sizeof_gr_complex, num_samples)
        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, filename)
        self.file_sink.set_unbuffered(False)

        self.connect(self.sdr, self.dc_block, self.head, self.file_sink)

# 실행
tb = IQ_Capture("capture_433MHz.bin", duration_sec=10.0)
tb.run()
print("캡처 완료")
```

### 7.2 캡처 데이터 로드 및 전처리

```python
import numpy as np

def load_iq(filename: str) -> np.ndarray:
    """GNU Radio file_sink 포맷 (complex64) 로드"""
    return np.fromfile(filename, dtype=np.complex64)

def normalize_iq(iq: np.ndarray) -> np.ndarray:
    """IQ 신호 정규화 (단위 전력)"""
    power = np.mean(np.abs(iq) ** 2)
    return iq / (np.sqrt(power) + 1e-9)

def iq_to_features(iq: np.ndarray, window=128) -> np.ndarray:
    """
    IQ → 딥러닝 입력 특징 변환
    shape: (N_windows, window, 2)  — 실수/허수 채널
    """
    n_windows = len(iq) // window
    iq_cut = iq[:n_windows * window].reshape(n_windows, window)
    features = np.stack([iq_cut.real, iq_cut.imag], axis=-1)
    return features.astype(np.float32)

# 사용 예시
iq_raw  = load_iq("capture_433MHz.bin")
iq_norm = normalize_iq(iq_raw)
X       = iq_to_features(iq_norm, window=128)
print(f"특징 형태: {X.shape}")   # (N, 128, 2)
```

---

## 8. 딥러닝 데이터셋 생성 파이프라인

실제 RF 수신 데이터 외에, GNU Radio 시뮬레이션으로 레이블된 데이터셋을 대량 생성한다.

```python
import numpy as np
import os

LABELS = {"BASK": 0, "BFSK": 1, "BPSK": 2}
SPS    = 200
FS     = 2e6
N_SAMPLES_PER_CLASS = 5000
WINDOW = 128
SNR_LIST = [-5, 0, 5, 10, 15, 20]

def generate_dataset(out_dir="dataset"):
    os.makedirs(out_dir, exist_ok=True)
    X_list, y_list = [], []

    for label, class_id in LABELS.items():
        print(f"[{label}] 데이터 생성 중...")
        for _ in range(N_SAMPLES_PER_CLASS):
            # 랜덤 비트 생성
            bits = np.random.randint(0, 2, size=64).astype(np.float32)

            # 변조
            if label == "BASK":
                sig = generate_bask(bits, sps=SPS, fc=100e3, fs=FS)
            elif label == "BFSK":
                sig = generate_bfsk(bits, sps=SPS, f0=80e3, f1=120e3, fs=FS)
            else:  # BPSK
                sig = generate_bpsk(bits, sps=SPS, fc=100e3, fs=FS)

            # 랜덤 SNR 추가
            snr = np.random.choice(SNR_LIST)
            sig_noisy = add_awgn(sig, snr_db=snr)

            # 창 분할 및 정규화
            sig_norm = normalize_iq(sig_noisy)
            windows  = iq_to_features(sig_norm, window=WINDOW)

            X_list.append(windows)
            y_list.extend([class_id] * len(windows))

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    # 셔플
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 저장
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    print(f"데이터셋 저장 완료: X={X.shape}, y={y.shape}")
    return X, y

# 실행
X, y = generate_dataset()
```

---

## 9. 딥러닝 모델 연동 — 실시간 분류

### 9.1 모델 구조 (CNN + LSTM 권장)

```python
# 요구사항: pip install torch
import torch
import torch.nn as nn

class ModulationClassifier(nn.Module):
    """
    IQ 시계열 분류 모델
    입력: (batch, window, 2) — [I채널, Q채널]
    출력: (batch, num_classes) — BASK / BFSK / BPSK
    """
    def __init__(self, window=128, num_classes=3):
        super().__init__()
        # CNN — 지역 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # LSTM — 시간적 의존성 학습
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, W, 2) → CNN 입력: (B, 2, W)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                   # (B, 128, W/4)
        x = x.permute(0, 2, 1)           # (B, W/4, 128)
        _, (h, _) = self.lstm(x)          # h: (2, B, 128)
        h = torch.cat([h[0], h[1]], dim=1) # (B, 256)
        return self.classifier(h)


# ─── 학습 루프 ─────────────────────────────────────────────────
def train_model(X, y, epochs=30, batch_size=256):
    from torch.utils.data import TensorDataset, DataLoader, random_split

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    n_val   = int(len(dataset) * 0.2)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = ModulationClassifier()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total   += len(yb)
        print(f"Epoch {ep+1:3d} | Val Acc: {correct/total*100:.2f}%")

    torch.save(model.state_dict(), "modulation_classifier.pth")
    return model
```

### 9.2 GNU Radio Embedded Python Block — 실시간 추론

GRC에서 RTL-SDR 수신 신호를 실시간으로 분류한다.

```python
# GRC Embedded Python Block 내부 코드
import numpy as np
from gnuradio import gr
import torch

CLASS_NAMES = ["BASK", "BFSK", "BPSK"]
WINDOW      = 128
MODEL_PATH  = "modulation_classifier.pth"

class blk(gr.sync_block):
    """실시간 변조 분류 블록"""

    def __init__(self, window=128, model_path=MODEL_PATH):
        gr.sync_block.__init__(self,
            name="DL Modulation Classifier",
            in_sig=[np.complex64],
            out_sig=[]             # 출력 없음 (메시지로 결과 송출)
        )
        self.window = window
        self.buffer = np.zeros(window, dtype=np.complex64)
        self.buf_idx = 0
        self.message_port_register_out(
            __import__('pmt').intern("class_out"))

        # 모델 로드
        from modulation_classifier_def import ModulationClassifier
        self.model = ModulationClassifier(window=window)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def work(self, input_items, output_items):
        inp = input_items[0]
        import pmt

        for sample in inp:
            self.buffer[self.buf_idx] = sample
            self.buf_idx += 1
            if self.buf_idx == self.window:
                # 정규화 및 추론
                sig = self.buffer.copy()
                sig /= (np.sqrt(np.mean(np.abs(sig)**2)) + 1e-9)
                feat = np.stack([sig.real, sig.imag], axis=-1)
                x    = torch.FloatTensor(feat).unsqueeze(0)  # (1,W,2)

                with torch.no_grad():
                    logits = self.model(x)
                    pred   = logits.argmax(1).item()
                    conf   = torch.softmax(logits, 1)[0, pred].item()

                label = CLASS_NAMES[pred]
                msg   = pmt.cons(pmt.intern(label),
                                 pmt.from_double(conf))
                self.message_port_pub(pmt.intern("class_out"), msg)
                self.buf_idx = 0

        return len(inp)
```

---

## 10. 데이터 복호화

분류된 변조 방식에 맞는 복조기를 적용하여 비트열을 복원하고, 문자열로 변환한다.

```python
class Demodulator:
    """분류 결과에 따라 적응적으로 복조"""

    def demodulate(self, iq: np.ndarray, mod_type: str,
                   sps: int = 200) -> np.ndarray:
        if mod_type == "BASK":
            return self._demod_bask(iq, sps)
        elif mod_type == "BFSK":
            return self._demod_bfsk(iq, sps)
        elif mod_type == "BPSK":
            return self._demod_bpsk(iq, sps)
        else:
            raise ValueError(f"알 수 없는 변조 방식: {mod_type}")

    def _demod_bask(self, iq, sps):
        """포락선 검파 (Envelope Detection)"""
        envelope = np.abs(iq)
        # 심볼별 평균
        n_sym  = len(envelope) // sps
        sym_e  = envelope[:n_sym*sps].reshape(n_sym, sps).mean(axis=1)
        threshold = (sym_e.max() + sym_e.min()) / 2
        return (sym_e > threshold).astype(np.float32)

    def _demod_bfsk(self, iq, sps):
        """주파수 판별기 (미분 검파)"""
        # 순시 주파수 = 위상 차분
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase)   # ∝ 순시 주파수
        n_sym = len(inst_freq) // sps
        sym_f = inst_freq[:n_sym*sps].reshape(n_sym, sps).mean(axis=1)
        threshold = 0.0
        return (sym_f > threshold).astype(np.float32)

    def _demod_bpsk(self, iq, sps):
        """코히어런트 검파 (상관 검파)"""
        # 캐리어 복구 (간이: 제곱 후 위상 절반)
        iq_sq   = iq ** 2
        carrier = np.angle(iq_sq) / 2
        derotated = iq * np.exp(-1j * carrier)
        # 심볼별 실수부 부호 판정
        n_sym = len(derotated) // sps
        sym_r = derotated[:n_sym*sps].reshape(n_sym, sps).mean(axis=1).real
        return (sym_r > 0).astype(np.float32)


# ─── 복호화 파이프라인 ─────────────────────────────────────────
demod = Demodulator()

def decode_signal(iq: np.ndarray, mod_type: str) -> str:
    """IQ 샘플 → 문자열 복원"""
    bits = demod.demodulate(iq, mod_type, sps=200)
    start = find_preamble(bits)
    if start == -1:
        return "[프리앰블 미검출]"
    data_bits = bits[start:start + 8 * 10]   # 최대 10문자
    return bits_to_string(data_bits)
```

---

## 11. 성능 측정

### 11.1 분류 정확도 측정

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(model, X_test, y_test):
    """분류 모델 성능 평가"""
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        y_pred = logits.argmax(1).numpy()

    acc = accuracy_score(y_test, y_pred)
    print(f"\n전체 정확도: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred,
                                  target_names=["BASK","BFSK","BPSK"]))

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["BASK","BFSK","BPSK"],
                yticklabels=["BASK","BFSK","BPSK"])
    plt.xlabel("예측"); plt.ylabel("실제")
    plt.title("변조 분류 혼동 행렬")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    return acc
```

### 11.2 SNR별 분류 성능 분석

```python
def accuracy_vs_snr(model, snr_range=range(-5, 25, 5)):
    """SNR 구간별 분류 정확도 측정 및 시각화"""
    import torch
    accs = {}
    for snr in snr_range:
        X_snr, y_snr = [], []
        for label, cid in LABELS.items():
            for _ in range(500):
                bits = np.random.randint(0, 2, 64).astype(np.float32)
                if label == "BASK":
                    sig = generate_bask(bits)
                elif label == "BFSK":
                    sig = generate_bfsk(bits)
                else:
                    sig = generate_bpsk(bits)
                sig_n = add_awgn(sig, snr_db=snr)
                feat  = iq_to_features(normalize_iq(sig_n), WINDOW)
                X_snr.append(feat[0])
                y_snr.append(cid)

        X_t = torch.FloatTensor(np.array(X_snr))
        with torch.no_grad():
            pred = model(X_t).argmax(1).numpy()
        accs[snr] = accuracy_score(np.array(y_snr), pred)
        print(f"SNR={snr:+3d} dB → 정확도: {accs[snr]*100:.1f}%")

    # 시각화
    plt.figure(figsize=(8, 4))
    plt.plot(list(accs.keys()), [v*100 for v in accs.values()],
             "o-", linewidth=2)
    plt.xlabel("SNR (dB)"); plt.ylabel("정확도 (%)")
    plt.title("SNR vs 변조 분류 정확도"); plt.grid(True)
    plt.axhline(90, color='r', linestyle='--', label='90% 기준선')
    plt.legend(); plt.tight_layout()
    plt.savefig("snr_vs_accuracy.png", dpi=150)
    return accs
```

### 11.3 비트 오류율 (BER) 및 전송 성공률

```python
def measure_ber(original_text: str, decoded_text: str) -> float:
    """
    BER (Bit Error Rate) 측정
    = 오류 비트 수 / 전체 비트 수
    """
    orig_bits = string_to_bits(original_text)
    # 길이 맞추기
    max_len  = max(len(orig_bits), len(string_to_bits(decoded_text)))
    dec_bits = string_to_bits(decoded_text[:len(original_text)])
    errors   = np.sum(orig_bits[:len(dec_bits)] != dec_bits)
    total    = len(orig_bits)
    return errors / total

def transmission_success_rate(results: list) -> dict:
    """
    전송 성공률 통계
    results: [{"sent": "BASK1", "received": "BASK1", "mod_correct": True}, ...]
    """
    total          = len(results)
    mod_correct    = sum(r["mod_correct"] for r in results)
    data_correct   = sum(r["sent"] == r["received"] for r in results)
    ber_values     = [measure_ber(r["sent"], r["received"]) for r in results]

    return {
        "total_packets"     : total,
        "mod_accuracy"      : mod_correct / total * 100,
        "data_success_rate" : data_correct / total * 100,
        "avg_ber"           : np.mean(ber_values),
        "min_ber"           : np.min(ber_values),
        "max_ber"           : np.max(ber_values),
    }
```

---

## 12. 전체 플로우그래프 코드

### 12.1 TX 플로우그래프 (HackRF)

```python
"""
tx_flowgraph.py
HackRF One을 이용한 다중 변조 신호 송신
"""
import numpy as np
from gnuradio import gr, blocks
from gnuradio.soapy import sink as soapy_sink

CENTER_FREQ = 433.1e6
SAMP_RATE   = 2e6
TX_GAIN     = 30
SPS         = 200
REPEAT      = True   # 신호 반복 송출

class TX_Flowgraph(gr.top_block):
    def __init__(self):
        super().__init__("Multi-Mod TX")

        # ── 복합 신호 생성 ────────────────────────────────────
        composite = make_composite_signal(fs=SAMP_RATE, sps=SPS)
        # AWGN 추가 (선택 — 채널 시뮬레이션 없이 실 RF로 보낼 경우 제거)
        # composite = add_awgn(composite, snr_db=15)

        composite_list = composite.tolist()

        # ── GNU Radio 블록 ─────────────────────────────────────
        self.vector_src = blocks.vector_source_c(
            composite_list, repeat=REPEAT)
        self.throttle = blocks.throttle(
            gr.sizeof_gr_complex, SAMP_RATE)

        # HackRF 싱크
        self.hackrf_sink = soapy_sink(
            "", "hackrf", 1, "", "", [""], [""])
        self.hackrf_sink.set_sample_rate(0, SAMP_RATE)
        self.hackrf_sink.set_frequency(0, CENTER_FREQ)
        self.hackrf_sink.set_gain(0, "AMP", 0)
        self.hackrf_sink.set_gain(0, "VGA", TX_GAIN)

        self.connect(self.vector_src, self.throttle, self.hackrf_sink)

if __name__ == "__main__":
    tb = TX_Flowgraph()
    tb.start()
    input("송신 중... Enter로 종료\n")
    tb.stop()
    tb.wait()
```

### 12.2 RX 플로우그래프 (RTL-SDR + 분류)

```python
"""
rx_flowgraph.py
RTL-SDR V4로 수신 → IQ 저장 + 실시간 분류
"""
import numpy as np
from gnuradio import gr, blocks, filter
from gnuradio.filter import firdes
from gnuradio.soapy import source as soapy_source

CENTER_FREQ = 433.1e6
SAMP_RATE   = 2.048e6
RX_GAIN     = 40

class RX_Flowgraph(gr.top_block):
    def __init__(self, save_file="rx_capture.bin"):
        super().__init__("Multi-Mod RX")

        # ── RTL-SDR 소스 ────────────────────────────────────────
        self.sdr = soapy_source("", "rtlsdr", 1, "", "", [""], [""])
        self.sdr.set_sample_rate(0, SAMP_RATE)
        self.sdr.set_frequency(0, CENTER_FREQ)
        self.sdr.set_gain(0, "TUNER", RX_GAIN)
        self.sdr.set_dc_offset_mode(0, True)
        self.sdr.set_iq_balance_mode(0, True)

        # ── DC 제거 ──────────────────────────────────────────────
        self.dc_block = blocks.dc_blocker_cc(32, True)

        # ── IQ 파일 저장 ─────────────────────────────────────────
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, save_file)

        # ── 딥러닝 분류기 (Embedded Python Block 방식) ──────────
        # 실제 GRC 에서는 Embedded Python Block 으로 삽입
        # self.classifier = DL_Classifier_Block()

        # ── 연결 ─────────────────────────────────────────────────
        self.connect(self.sdr, self.dc_block)
        self.connect(self.dc_block, self.file_sink)
        # self.connect(self.dc_block, self.classifier)

if __name__ == "__main__":
    tb = RX_Flowgraph("rx_capture.bin")
    tb.start()
    input("수신 중... Enter로 종료\n")
    tb.stop()
    tb.wait()
    print("수신 완료. 파일 저장됨: rx_capture.bin")
```

---

## 13. GRC 블록 구성 참고

### 13.1 TX GRC 블록 구성

```
[Options]
  ID: tx_multi_mod
  Output Language: Python

[Variable]
  samp_rate = 2e6
  center_freq = 433.1e6
  sps = 200

[Vector Source] ── composite IQ 데이터
  Type: complex
  Repeat: Yes

[Throttle]
  Sample Rate: samp_rate

[Soapy HackRF Sink]
  Dev Args: ""
  Sample Rate: samp_rate
  Center Freq: center_freq
  Bandwidth: samp_rate
  Gain: VGA=30
```

### 13.2 RX GRC 블록 구성

```
[Soapy RTLSDR Source]
  Sample Rate: 2.048e6
  Center Freq: 433.1e6
  Gain: TUNER=40

[DC Blocker]
  Length: 32

──────────────────── 분기 ────────────────────────
분기 A: [File Sink] → rx_capture.bin
분기 B: [Embedded Python Block] → 실시간 분류
         └─ Message → [Message Debug] (분류 결과 출력)

분기 C (채널별 분리):
  [Freq Xlating FIR Filter] (offset=-300kHz) → [BASK 복조기]
  [Freq Xlating FIR Filter] (offset=+100kHz) → [BFSK 복조기]
  [Freq Xlating FIR Filter] (offset=+300kHz) → [BPSK 복조기]
```

### 13.3 필수 GNU Radio 블록 목록

| 블록 이름 | 카테고리 | 용도 |
|-----------|----------|------|
| `Soapy RTLSDR Source` | Soapy SDR | RTL-SDR 수신 |
| `Soapy HackRF Sink` | Soapy SDR | HackRF 송신 |
| `DC Blocker` | Filters | DC 오프셋 제거 |
| `Freq Xlating FIR Filter` | Filters | 채널 선택 + 다운샘플 |
| `File Sink` | File Operators | IQ 파일 저장 |
| `Vector Source` | Sources | 미리 생성한 신호 재생 |
| `Throttle` | Misc | 시뮬레이션 속도 제한 |
| `Channel Model` | Channels | AWGN 추가 |
| `Add` | Math Operators | 다중 신호 합산 |
| `Multiply Const` | Math Operators | 게인 조절 |
| `Embedded Python Block` | Python | 딥러닝 추론 |
| `QT GUI Frequency Sink` | GUI Widgets | 스펙트럼 시각화 |
| `QT GUI Time Sink` | GUI Widgets | 파형 시각화 |
| `QT GUI Waterfall Sink` | GUI Widgets | 시간-주파수 시각화 |

---

## 참고 자료

- [GNU Radio 공식 튜토리얼](https://wiki.gnuradio.org/index.php/Tutorials)
- [GNU Radio Doxygen API](https://www.gnuradio.org/doc/doxygen/)
- [RTL-SDR Blog V4 드라이버](https://github.com/rtlsdrblog/rtl-sdr-blog)
- [HackRF One 공식 문서](https://hackrf.readthedocs.io)
- [RadioML 데이터셋 논문 (변조 분류 딥러닝 참고)](https://arxiv.org/abs/1602.04105)
- [Your First Flowgraph](https://wiki.gnuradio.org/index.php/Your_First_Flowgraph)
- [Embedded Python Block](https://wiki.gnuradio.org/index.php/Embedded_Python_Block)
- [GNU Radio Channel Model](https://wiki.gnuradio.org/index.php/Channel_Model)
