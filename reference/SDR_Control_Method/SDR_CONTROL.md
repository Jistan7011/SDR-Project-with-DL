# SDR 제어 방법 및 예제 코드

이 문서는 HackRF One 송신기와 RTL-SDR Blog V4 수신기를 사용해 OTA RF 신호를 송수신하고, BASK/BFSK/BPSK 변조 데이터 수집에 필요한 기능을 다루는 방법을 정리한 문서이다.

실험 결과 해석이나 AI 모델 학습 방법이 아니라, SDR 장비 제어와 송수신 코드 사용법에 집중한다.

---

## 1. 하드웨어 구성

사용 장비:

| 장비 | 역할 |
| --- | --- |
| HackRF One | RF 송신기, TX |
| RTL-SDR Blog V4 | RF 수신기, RX |
| PC | 신호 생성, SDR 제어, IQ 저장 |

OTA 구성:

```text
HackRF One antenna  ← 약 1 m OTA →  RTL-SDR Blog V4 antenna
```

기본 연결:

```text
PC USB ─ HackRF One
PC USB ─ RTL-SDR Blog V4
```

주의:

```text
HackRF와 RTL-SDR을 동시에 사용하는 프로그램이 있으면 장비 open이 실패할 수 있다.
SDR#, SDR++, CubicSDR 같은 프로그램은 실험 전에 종료한다.
```

---

## 2. 실행 환경 구조

이 프로젝트는 두 Python 환경을 함께 사용한다.

| 환경 | 용도 |
| --- | --- |
| `D:\ai_projects\SDR\.venv` | 일반 Python 실행, 데이터 처리, 분석 스크립트 실행 |
| `C:\Users\qus70\radioconda` | SoapySDR, HackRF, RTL-SDR 장비 제어 |

일반적으로 사용자는 `.venv`에서 명령을 실행한다.

```powershell
cd D:\ai_projects\SDR
.\.venv\Scripts\Activate.ps1
```

이후 Oshea2018 sourcecode로 이동한다.

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode

$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.fixed.yaml"
```

SDR 장비를 실제로 여는 코드는 내부에서 RadioConda를 호출한다.

```text
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m ...
```

---

## 3. SDR 장비 인식 확인

SoapySDR에서 HackRF와 RTL-SDR이 보이는지 확인한다.

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode

$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"

cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose
```

정상 로그 예:

```text
SoapySDR devices:
  [0] {... rtlsdr ...}
  [1] {... hackrf ...}
open driver=rtlsdr: ok
open driver=hackrf: ok
```

RTL-SDR 정상 로그 예:

```text
Found Rafael Micro R828D tuner
RTL-SDR Blog V4 Detected
[INFO] Opening Generic RTL2832U OEM :: 00000001...
```

HackRF 정상 로그 예:

```text
[INFO] Opening HackRF One #0 ...
```

장비를 못 찾는 경우:

```text
No RTL-SDR devices found
Could not open HackRF via SoapySDR
```

대응:

```text
1. HackRF / RTL-SDR USB 재연결
2. SDR# / SDR++ 등 장비 점유 프로그램 종료
3. PowerShell 새로 열기
4. USB 포트 변경
5. PC 재부팅
```

---

## 4. RTL-SDR 단독 수신

RTL-SDR로 일정 시간 IQ를 저장하는 기본 명령이다.

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode

$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.fixed.yaml"

cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.capture_iq `
  --config $CFG `
  --output ..\data\manual_rx\noise_only.bin `
  --seconds 2.0 `
  --sample-rate 2400000 `
  --rx-gain 30
```

저장 형식:

```text
complex64 binary
I/Q interleaved가 아니라 numpy complex64 배열을 그대로 tofile 저장
```

정상 출력:

```text
Captured 4800000 complex64 samples to ..\data\manual_rx\noise_only.bin
```

샘플 수 계산:

```text
sample_rate × seconds = 저장 complex sample 수
2.4e6 × 2.0 = 4,800,000 complex64 samples
```

---

## 5. HackRF 단독 송신

HackRF로 특정 변조 신호를 송신한다.

지원 변조:

```text
BASK
BFSK
BPSK
```

BASK 송신 예:

```powershell
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.hackrf_tx_oshea2018 `
  --config $CFG `
  --modulation BASK `
  --seconds 5.0 `
  --seed 42 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --baseband-offset-hz 500000
```

BFSK 송신 예:

```powershell
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.hackrf_tx_oshea2018 `
  --config $CFG `
  --modulation BFSK `
  --seconds 5.0 `
  --seed 43 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --baseband-offset-hz 500000
```

BPSK 송신 예:

```powershell
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.hackrf_tx_oshea2018 `
  --config $CFG `
  --modulation BPSK `
  --seconds 5.0 `
  --seed 44 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --baseband-offset-hz 500000
```

정상 로그:

```text
TX Oshea2018 modulation=BPSK offset_hz=500000.0 amp=0.0 vga=30.0
```

중요:

```text
tx_amp_gain=0은 송신 off가 아니다.
현재 송신 세기는 주로 tx_vga_gain으로 조절한다.
```

---

## 6. 변조별 송신 신호 생성 방식

현재 송신 코드는 다음 파일을 사용한다.

```text
D:\ai_projects\SDR\experiments\oshea2018\sourcecode\src\signal\oshea2018_waveform.py
```

### 6.1 Random payload

각 capture는 random bit sequence를 사용한다.

```python
def random_bits(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(0, 2, size=count, dtype=np.uint8)
```

목적:

```text
모델이 payload pattern을 외우지 못하게 한다.
변조 label과 bit pattern의 상관관계를 제거한다.
```

### 6.2 BASK

BASK는 bit 값에 따라 amplitude가 바뀐다.

```text
bit 0 -> low amplitude
bit 1 -> high amplitude
```

현재 구현:

```python
symbols = np.where(bit_values > 0.5, 1.0, bask_low_amplitude)
```

기본 `bask_low_amplitude`:

```text
0.15
```

핵심 특징:

```text
magnitude
envelope
amplitude variation
```

### 6.3 BFSK

BFSK는 bit 값에 따라 주파수가 바뀐다.

```text
bit 0 -> -freq_dev
bit 1 -> +freq_dev
```

현재 구현:

```python
freqs = np.repeat(np.where(bit_values > 0.5, bfsk_freq_dev_hz, -bfsk_freq_dev_hz), sps)
phase = np.cumsum(2.0 * np.pi * freqs / sample_rate)
iq = np.exp(1j * phase)
```

기본 frequency deviation:

```text
bfsk_freq_dev_hz: 50000
```

핵심 특징:

```text
instantaneous frequency
phase slope
frequency transition
```

### 6.4 BPSK

BPSK는 bit 값에 따라 phase가 0 또는 pi로 바뀐다.

```text
bit 0 -> +1
bit 1 -> -1
```

현재 구현:

```python
symbols = np.where(bit_values > 0.5, -1.0, 1.0)
```

핵심 특징:

```text
phase transition
differential phase
constellation sign flip
```

---

## 7. 송신 offset 사용

HackRF는 baseband IQ를 RF 중심 주파수에 올려 송신한다.  
RTL-SDR은 중심 주파수 근처에 DC spike가 생길 수 있으므로, 신호를 중심에서 떨어뜨려 송신한다.

예:

```text
center_freq = 433.92 MHz
baseband_offset_hz = 500 kHz
실제 신호 중심 = 433.92 MHz + 500 kHz
```

송신 코드에서 offset 적용:

```python
n = np.arange(len(iq), dtype=np.float64)
iq = iq * np.exp(1j * 2.0 * np.pi * offset * n / sample_rate)
```

권장 offset:

```text
250 kHz
500 kHz
```

---

## 8. RX/TX 동시 실행

실제 OTA 데이터 수집은 RX를 먼저 켜고, 잠시 후 TX를 켠다.

이유:

```text
송신 시작 구간을 놓치지 않기 위해서
TX 전 noise-only 또는 RX lead 구간을 확보하기 위해서
```

수동으로 두 터미널에서 실행할 수도 있지만, 권장 방식은 세션 수집 스크립트를 쓰는 것이다.

```powershell
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config $CFG `
  --session-id session_001 `
  --output-root ..\data\manual_session `
  --captures-per-class 1 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --rx-gain 30 `
  --baseband-offset-hz 500000 `
  --max-retries 3
```

이 명령이 내부적으로 하는 일:

```text
1. noise_only.bin 수집
2. BASK capture 수집
3. BFSK capture 수집
4. BPSK capture 수집
5. 각 capture마다 품질 계산
6. metadata.json 저장
```

---

## 9. 세션 수집 스크립트 내부 동작

세션 수집 함수:

```text
src.experiment.run_oshea2018_capture_session
```

핵심 동작:

```text
RTL-SDR RX 프로세스 시작
rx_lead_seconds 만큼 대기
HackRF TX 프로세스 시작
TX 종료 대기
RX 종료 대기
capture 품질 계산
품질 실패 시 재시도
```

관련 설정:

```yaml
ota:
  capture_seconds: 5.0
  noise_capture_seconds: 2.0
  rx_lead_seconds: 0.5
  active_start_seconds: 1.1
  active_duration_seconds: 3.8
```

의미:

| 설정 | 의미 |
| --- | --- |
| `capture_seconds` | RX 전체 수신 시간 |
| `noise_capture_seconds` | TX off noise-only 수신 시간 |
| `rx_lead_seconds` | RX 시작 후 TX 시작까지 대기 시간 |
| `active_start_seconds` | 품질 판단/학습에 사용할 안정 구간 시작 |
| `active_duration_seconds` | 안정 구간 길이 |

---

## 10. Noise-only 수집

noise-only는 TX를 켜지 않고 RTL-SDR만 수신한 데이터다.

용도:

```text
noise floor 측정
주변 간섭 확인
TX 신호 대비 RMS ratio 계산
SNR 추정
quality gate 기준
```

세션 수집 스크립트는 각 session마다 자동으로 수집한다.

저장 파일:

```text
noise_only.bin
```

수동 수집 예:

```powershell
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.capture_iq `
  --config $CFG `
  --output ..\data\manual_rx\noise_only.bin `
  --seconds 2.0 `
  --sample-rate 2400000 `
  --rx-gain 30
```

---

## 11. Capture 품질 확인

세션 수집 후 다음과 같은 품질 로그가 출력된다.

```text
QUALITY: {
  "noise_rms": ...,
  "tx_rms": ...,
  "tx_to_noise_rms_ratio": ...,
  "estimated_snr_db": ...,
  "spectral_peak_prominence": ...,
  "clipping_rate": ...,
  "quality_pass": true,
  "quality_reason": "pass"
}
```

주요 항목:

| 항목 | 의미 |
| --- | --- |
| `noise_rms` | noise-only 구간 RMS |
| `tx_rms` | TX active 구간 RMS |
| `tx_to_noise_rms_ratio` | TX 신호가 noise보다 얼마나 큰지 |
| `estimated_snr_db` | 추정 SNR |
| `spectral_peak_prominence` | FFT peak가 얼마나 두드러지는지 |
| `clipping_rate` | I/Q clipping 의심 비율 |
| `quality_pass` | 품질 gate 통과 여부 |

권장 판단:

```text
tx_to_noise_rms_ratio는 충분히 커야 한다.
estimated_snr_db는 너무 낮으면 안 된다.
clipping_rate는 낮아야 한다.
quality_pass=false가 반복되면 gain/offset/안테나 조건을 바꾼다.
```

---

## 12. Capture 분석

저장된 `.bin` 파일의 RMS와 spectrum을 확인할 수 있다.

```powershell
& $PY -m src.sdr.analyze_capture `
  --config $CFG `
  --input ..\data\manual_session\session_001\session_001_bpsk_000.bin
```

생성 결과:

```text
results/real_capture_analysis/*_rms.png
results/real_capture_analysis/*_spectrum.png
results/real_capture_analysis/*_summary.json
```

확인할 것:

```text
RMS가 TX active 구간에서 올라가는가
FFT peak가 baseband offset 근처에 있는가
신호가 clipping되어 이상하게 넓게 퍼지지 않았는가
```

---

## 13. Channelize / Downsample

수신 raw IQ는 넓은 대역이다.  
실제 송신 신호는 baseband offset 위치에 있으므로, 학습/품질 분석 전에 해당 신호를 baseband로 옮기고 downsample한다.

사용 함수:

```python
from src.signal.channelize import channelize_and_downsample
```

예제 코드:

```python
import numpy as np
from src.signal.channelize import channelize_and_downsample

raw_iq = np.fromfile("capture.bin", dtype=np.complex64)

channel_iq, effective_sample_rate = channelize_and_downsample(
    raw_iq,
    sample_rate=2_400_000,
    channel_center_hz=500_000,
    channel_bandwidth_hz=100_000,
    target_sample_rate=160_000,
)

print(channel_iq.shape)
print(effective_sample_rate)
```

의미:

```text
sample_rate: 원본 RTL-SDR 수신 sample rate
channel_center_hz: 송신 offset
channel_bandwidth_hz: 남길 대역폭
target_sample_rate: downsample 후 sample rate
```

---

## 14. 직접 Python으로 수신하기

`capture_iq.py`의 핵심 구조는 다음과 같다.

```python
from src.sdr.soapy_common import require_soapy

SoapySDR, SOAPY_SDR_CF32, SOAPY_SDR_RX, _ = require_soapy()

sdr = SoapySDR.Device("driver=rtlsdr")
sdr.setSampleRate(SOAPY_SDR_RX, 0, 2_400_000)
sdr.setFrequency(SOAPY_SDR_RX, 0, 433_920_000)
sdr.setGain(SOAPY_SDR_RX, 0, "TUNER", 30)

stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(stream)
```

수신 loop:

```python
import numpy as np

total = int(2_400_000 * 2.0)
data = np.empty(total, dtype=np.complex64)
offset = 0

while offset < total:
    chunk = np.empty(min(4096, total - offset), dtype=np.complex64)
    sr = sdr.readStream(stream, [chunk], len(chunk))
    if sr.ret > 0:
        data[offset : offset + sr.ret] = chunk[: sr.ret]
        offset += sr.ret
    elif sr.ret < 0:
        raise RuntimeError(f"readStream failed: {sr.ret}")

sdr.deactivateStream(stream)
sdr.closeStream(stream)
data.tofile("capture.bin")
```

---

## 15. 직접 Python으로 송신하기

`hackrf_tx_oshea2018.py`의 핵심 구조는 다음과 같다.

```python
from src.sdr.soapy_common import require_soapy

SoapySDR, SOAPY_SDR_CF32, _, SOAPY_SDR_TX = require_soapy()

sdr = SoapySDR.Device("driver=hackrf")
sdr.setSampleRate(SOAPY_SDR_TX, 0, 2_400_000)
sdr.setFrequency(SOAPY_SDR_TX, 0, 433_920_000)
sdr.setGain(SOAPY_SDR_TX, 0, "AMP", 0)
sdr.setGain(SOAPY_SDR_TX, 0, "VGA", 30)
```

송신 loop:

```python
stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(stream)

index = 0
while transmitting:
    chunk = iq[index : index + 4096]
    if len(chunk) == 0:
        index = 0
        continue
    sdr.writeStream(stream, [chunk], len(chunk))
    index += len(chunk)

sdr.deactivateStream(stream)
sdr.closeStream(stream)
```

---

## 16. 변조 신호 생성 예제

아래 코드는 BPSK baseband IQ를 생성하는 예시다.

```python
import numpy as np
from src.signal.oshea2018_waveform import random_bits, generate_clean_modulation

rng = np.random.default_rng(42)
sample_rate = 2_400_000
symbol_rate = 5_000
seconds = 5.0

bits = random_bits(rng, int(seconds * symbol_rate))

iq = generate_clean_modulation(
    "BPSK",
    bits,
    sample_rate=sample_rate,
    symbol_rate=symbol_rate,
    rolloff=0.35,
    bfsk_freq_dev_hz=50_000,
)

print(iq.dtype, iq.shape)
```

송신 offset 적용:

```python
offset_hz = 500_000
n = np.arange(len(iq), dtype=np.float64)
iq = iq * np.exp(1j * 2.0 * np.pi * offset_hz * n / sample_rate)
iq = (iq / (np.max(np.abs(iq)) + 1e-9) * 0.5).astype(np.complex64)
```

---

## 17. 세션 metadata

세션 수집 후 `metadata.json`이 저장된다.

주요 항목:

```json
{
  "experiment": "Oshea2018",
  "session_id": "session_001",
  "center_freq": 433920000,
  "sample_rate": 2400000,
  "symbol_rate": 5000,
  "distance_m": 1.0,
  "baseband_offset_hz": 500000,
  "tx_vga_gain": 30,
  "tx_amp_gain": 0,
  "rx_gain": 30,
  "agc": false,
  "noise_only_file": "...",
  "random_payload_bits": true,
  "captures": []
}
```

각 capture에는 다음 정보가 들어간다.

```text
file
modulation
capture_index
payload_seed
random_payload_bits
quality
```

---

## 18. Gain 조정 가이드

송신 gain:

```text
tx_vga_gain: HackRF 송신 세기 조절의 핵심 값
tx_amp_gain: 보통 0 유지
```

수신 gain:

```text
rx_gain: RTL-SDR 수신 gain
agc: false 권장
```

신호가 약할 때:

```text
tx_vga_gain 증가
rx_gain 증가
안테나 방향 조정
거리 줄이기
offset 변경
```

신호가 너무 강할 때:

```text
tx_vga_gain 감소
rx_gain 감소
안테나 방향 살짝 틀기
거리 늘리기
attenuator 사용
```

clipping 의심:

```text
clipping_rate가 높음
I/Q 값이 최대/최소 근처에 붙음
spectrum이 비정상적으로 넓게 퍼짐
변조별 차이가 사라짐
```

---

## 19. 권장 점검 순서

SDR 실험 전에 다음 순서로 확인한다.

```text
1. SoapySDR 장비 인식 확인
2. RTL-SDR noise-only 수집
3. HackRF 단일 변조 짧게 송신
4. RX/TX 동시 capture 1회 수행
5. QUALITY 로그 확인
6. analyze_capture로 RMS / spectrum 확인
7. gain과 offset 조정
```

최소 점검 명령:

```powershell
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config $CFG `
  --session-id session_001 `
  --output-root ..\data\hardware_check `
  --captures-per-class 1 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --rx-gain 30 `
  --baseband-offset-hz 500000 `
  --max-retries 1
```

---

## 20. 자주 생기는 문제

### RTL-SDR이 열리지 않음

로그:

```text
Could not open RTL-SDR via SoapySDR
No RTL-SDR devices found
```

대응:

```text
USB 재연결
다른 SDR 프로그램 종료
장치 관리자 확인
PowerShell 새로 열기
```

### HackRF가 열리지 않음

로그:

```text
Could not open HackRF via SoapySDR
```

대응:

```text
HackRF USB 재연결
다른 송신 프로그램 종료
USB 포트 변경
```

### 송신은 되는데 수신 신호가 약함

확인:

```text
tx_to_noise_rms_ratio 낮음
estimated_snr_db 낮음
FFT peak가 약함
```

대응:

```text
tx_vga_gain 증가
rx_gain 증가
안테나 방향 조정
offset 변경
```

### BFSK만 잘 보이고 BASK/BPSK가 약함

가능 원인:

```text
BFSK는 spectral peak가 강하게 보이기 쉬움
BASK/BPSK는 amplitude/phase evidence가 gain, CFO, phase drift에 민감함
수신 조건이 BFSK에만 유리할 수 있음
```

대응:

```text
BASK/BPSK 품질을 별도로 확인
BPSK differential phase 확인
gain을 낮추거나 높이며 clipping과 SNR 균형 확인
symbol_rate 조정
```

---

## 21. 안전 주의

HackRF는 송신 장비이므로 주변 RF 환경과 법규를 고려해야 한다.

권장:

```text
낮은 TX gain으로 시작
짧은 시간 송신
필요 이상으로 강하게 송신하지 않기
주변 신호와 간섭 확인
실험 주파수 사용 가능 여부 확인
```

실험 중에는 다음을 피한다.

```text
불필요한 장시간 송신
높은 gain에서 반복 송신
허가되지 않은 주파수 대역 사용
안테나 없이 HackRF 송신
```
