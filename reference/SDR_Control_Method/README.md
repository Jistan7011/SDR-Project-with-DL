# HackRF One + RTL-SDR 송수신 제어 방법

이 폴더는 HackRF One 송신기와 RTL-SDR Blog V4 수신기를 사용해 OTA IQ 데이터를 송수신하는 최소 절차와 예제 코드를 정리한다.

논문 재현이나 AI 학습 절차가 아니라, SDR 장비 제어와 데이터 송수신 확인에만 집중한다.

## 기준 환경

| 용도 | 환경 | 경로 |
| --- | --- | --- |
| SDR 장비 제어 | RadioConda | `C:\Users\qus70\radioconda` |
| 일반 분석/테스트 | 프로젝트 venv | `D:\ai_projects\SDR\.venv` |

SDR 장비를 실제로 여는 명령은 RadioConda에서 실행한다.

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python ..."
```

## 하드웨어 구성

```text
PC USB -> HackRF One -> antenna  ~ OTA ~  antenna <- RTL-SDR Blog V4 <- PC USB
```

주의:

- SDR#, SDR++, CubicSDR처럼 장비를 점유하는 프로그램은 먼저 종료한다.
- HackRF One은 송신 장비이므로 주변 RF 환경과 법규를 고려한다.
- 테스트 기본값은 `tx_amp_gain=0`, `tx_vga_gain=30`, `rx_gain=30`이다.
- 신호는 RTL-SDR DC spike를 피하기 위해 RF center에서 `baseband_offset_hz`만큼 떨어뜨려 송신한다.

## 이번 테스트에서 확인한 성공 경로

2026-05-10 테스트 기준:

```text
HackRF One: hackrf_info 인식 OK
RTL-SDR Blog V4: SoapySDR open OK
HackRF One: hackrf_transfer backend 단독 TX OK
RTL-SDR: SoapySDR complex64 capture OK
RX/TX 동시 capture: BASK/BFSK/BPSK 모두 quality_pass true
```

성공한 동시 capture quality:

```text
BASK: ratio 3.826, SNR 11.35 dB, clipping 0.0
BFSK: ratio 3.421, SNR 10.30 dB, clipping 0.0
BPSK: ratio 4.045, SNR 11.86 dB, clipping 0.0
```

## 권장 송수신 방식

### RX

RTL-SDR는 SoapySDR로 열고 complex64 IQ를 그대로 파일에 저장한다.

```text
Soapy RTL-SDR source -> numpy complex64 -> .bin
```

### TX

HackRF One은 종료가 확실한 `hackrf_transfer` 기반 송신을 기본으로 쓴다.

```text
generated complex64 IQ -> interleaved CS8 file -> hackrf_transfer -t ... -n ... -R
```

이 방식은 송신 sample 수(`-n`)가 정해져 있어 테스트 자동화에 적합하다.

### 비권장 기본값

Soapy direct TX와 GNU Radio/osmosdr TX는 fallback으로만 둔다. 단독 짧은 송신은 가능해도, 긴 waveform 또는 동시 capture에서 종료 지연/timeout이 발생할 수 있었다.

## 빠른 실행

PowerShell에서:

```powershell
cd D:\ai_projects\SDR\SDR_Control_Method
.\examples\run_rx_tx_smoke_test.ps1
```

이 스크립트는 다음을 순서대로 실행한다.

1. `hackrf_info`
2. SoapySDR 장비 open 확인
3. RTL-SDR noise-only capture
4. HackRF BPSK 단독 송신
5. RX/TX 동시 BASK/BFSK/BPSK capture
6. noise 대비 capture RMS/SNR proxy 분석

결과는 다음 폴더에 저장된다.

```text
SDR_Control_Method/output/smoke_test
```

## 개별 예제

### 1. 장비 확인

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && hackrf_info"
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python examples\diagnose_soapy_devices.py"
```

### 2. RTL-SDR 단독 수신

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python examples\rx_capture_soapy.py --output output\noise_only.bin --seconds 2 --center-freq 433920000 --sample-rate 2400000 --rx-gain 30"
```

저장 형식:

```text
numpy complex64 binary
sample count = sample_rate * seconds
file size = sample count * 8 bytes
```

### 3. HackRF 단독 송신

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python examples\tx_hackrf_transfer.py --modulation BPSK --seconds 1 --center-freq 433920000 --sample-rate 2400000 --symbol-rate 5000 --baseband-offset-hz 500000 --tx-vga-gain 30 --tx-amp-gain 0 --seed 42"
```

### 4. RX/TX 동시 capture

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python examples\rx_tx_capture_once.py --output-dir output\one_capture --modulation BPSK --seconds 5 --tx-seconds 4.5 --rx-lead-seconds 0.5 --center-freq 433920000 --sample-rate 2400000 --symbol-rate 5000 --baseband-offset-hz 500000 --rx-gain 30 --tx-vga-gain 30 --tx-amp-gain 0 --seed 42"
```

### 5. capture 분석

```powershell
D:\ai_projects\SDR\.venv\Scripts\python.exe examples\analyze_iq_capture.py --noise output\one_capture\noise_only.bin --capture output\one_capture\bpsk_capture.bin --sample-rate 2400000 --active-start-seconds 1.1 --active-duration-seconds 3.8
```

## 파일 구성

```text
SDR_Control_Method/
  README.md
  examples/
    diagnose_soapy_devices.py
    rx_capture_soapy.py
    tx_hackrf_transfer.py
    rx_tx_capture_once.py
    analyze_iq_capture.py
    run_rx_tx_smoke_test.ps1
```

## 정상/실패 판단

장비 제어 레벨의 정상 기준:

```text
HackRF One과 RTL-SDR가 모두 open된다.
noise_only.bin이 지정 sample 수만큼 저장된다.
HackRF TX가 지정 시간 이후 종료된다.
RX/TX 동시 capture 파일이 생성된다.
metadata.json이 생성된다.
capture RMS가 noise RMS보다 충분히 크다.
clipping이 거의 없다.
테스트 종료 후 radioconda python/cmd/hackrf 프로세스가 남지 않는다.
```

데이터셋 수집으로 넘어가기 전 기준:

```text
BASK/BFSK/BPSK 각각 1회 이상 동시 capture가 정상 종료
각 capture의 noise 대비 RMS ratio > 2
estimated SNR proxy > 6 dB
metadata에 RF 조건, payload seed, file path 기록
```

