# SDR / AI 실행 환경 설정

이 문서는 새 세션에서 SDR 송수신 환경과 AI 학습 환경을 다시 잡기 위한 환경 설정 문서이다.  
실험 절차, 모델 비교, 데이터 수집 계획은 다루지 않고, 실행 환경을 복원하는 데 필요한 정보만 정리한다.

---

## 1. 프로젝트 기본 경로

프로젝트 루트:

```text
D:\ai_projects\SDR
```

PowerShell에서 이동:

```powershell
cd D:\ai_projects\SDR
```

주요 가상환경:

```text
D:\ai_projects\SDR\.venv
```

SDR 제어용 RadioConda:

```text
C:\Users\qus70\radioconda
```

---

## 2. 전체 환경 구조

이 프로젝트는 Python 환경을 두 개로 나누어 사용한다.

```text
AI / 데이터 처리 / 학습:
  D:\ai_projects\SDR\.venv

SDR 하드웨어 제어:
  C:\Users\qus70\radioconda
```

역할 구분:

| 환경 | 역할 |
| --- | --- |
| `.venv` | PyTorch 학습, 데이터셋 생성, 평가, 분석 코드 실행 |
| `radioconda` | HackRF One 송신, RTL-SDR Blog V4 수신, SoapySDR 기반 장비 제어 |

주의할 점:

```text
딥러닝 학습은 .venv에서 실행한다.
SDR 장비를 직접 여는 코드는 radioconda에서 실행한다.
일반 실험 스크립트는 .venv에서 시작하지만, 내부적으로 radioconda를 호출해 SDR을 제어한다.
```

---

## 3. AI 실행 환경

### 3.1 가상환경 활성화

PowerShell에서:

```powershell
cd D:\ai_projects\SDR
.\.venv\Scripts\Activate.ps1
```

정상 상태:

```text
(.venv) PS D:\ai_projects\SDR>
```

실행 정책 문제로 activation이 막히면 현재 PowerShell 세션에서만 허용한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3.2 Python 경로 변수

PowerShell에서 자주 쓰는 변수:

```powershell
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
```

Python 확인:

```powershell
& $PY --version
& $PY -m pip --version
```

현재 실행 중인 Python이 `.venv`인지 확인:

```powershell
where.exe python
```

정상 예시:

```text
D:\ai_projects\SDR\.venv\Scripts\python.exe
```

### 3.3 PyTorch / CUDA 확인

GPU 사용 가능 여부 확인:

```powershell
& $PY -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

정상 판단:

```text
cuda available: True
```

GPU 상태 확인:

```powershell
nvidia-smi
```

확인할 항목:

```text
GPU 이름
GPU 메모리 사용량
python.exe 프로세스 존재 여부
GPU-Util
```

학습 중인데 GPU 사용률이 낮으면 다음을 의심한다.

```text
데이터 로딩 병목
CPU 전처리 병목
num_workers 설정 부족
batch_size가 너무 작음
실제로 CPU device로 실행 중
```

---

## 4. Python 패키지 관리 원칙

패키지는 반드시 프로젝트 `.venv`에 설치한다.

권장 방식:

```powershell
& $PY -m pip install 패키지명
```

피해야 할 방식:

```powershell
pip install 패키지명
```

이유:

```text
pip만 쓰면 전역 Python 또는 다른 환경에 설치될 수 있다.
항상 python -m pip 형태로 현재 Python에 설치되도록 한다.
```

설치된 패키지 확인:

```powershell
& $PY -m pip list
& $PY -m pip show torch
& $PY -m pip show numpy
& $PY -m pip show scipy
& $PY -m pip show scikit-learn
```

XGBoost 확인:

```powershell
& $PY -m pip show xgboost
```

Python에서 import 확인:

```powershell
& $PY -c "import numpy, scipy, sklearn; print('numpy/scipy/sklearn ok')"
& $PY -c "import xgboost; print('xgboost ok')"
```

---

## 5. 코드 실행용 PYTHONPATH

실험 코드가 `src.*` 형태의 모듈 import를 사용하므로, 실행 위치에 따라 `PYTHONPATH` 설정이 필요하다.

Oshea2018 코드 기준:

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
```

확인:

```powershell
echo $env:PYTHONPATH
```

이후 Python 모듈은 다음 형태로 실행한다.

```powershell
& $PY -m src.some.module
```

---

## 6. SDR 하드웨어 구성

사용 장비:

| 장비 | 역할 |
| --- | --- |
| HackRF One | 송신기, TX |
| RTL-SDR Blog V4 | 수신기, RX |
| PC | 신호 생성, SDR 제어, IQ 저장, AI 학습 |

물리 구성:

```text
PC
├── HackRF One USB 연결
└── RTL-SDR Blog V4 USB 연결

HackRF One antenna  ← 1 m OTA →  RTL-SDR Blog V4 antenna
```

OTA 조건:

```text
송수신 방식: RF cable 없음, 안테나 간 무선 송수신
거리: 약 1 m
TX: HackRF One
RX: RTL-SDR Blog V4
```

주의:

```text
1 m 거리는 신호가 너무 강하거나 약할 수 있다.
TX gain, RX gain, 안테나 방향에 따라 clipping 또는 수신 실패가 발생할 수 있다.
RTL-SDR AGC는 BASK amplitude 정보를 흐릴 수 있으므로 끄는 것이 좋다.
```

---

## 7. RadioConda 역할

SDR 장비 제어는 `radioconda` 환경에서 실행한다.

RadioConda 경로:

```text
C:\Users\qus70\radioconda
```

RadioConda가 필요한 이유:

```text
SoapySDR
HackRF driver
RTL-SDR driver
SDR Python binding
```

이 프로젝트에서는 보통 사용자가 RadioConda를 직접 활성화하지 않는다.  
대신 `.venv`에서 실행한 Python 스크립트가 내부적으로 다음 형식의 명령어를 만든다.

```text
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m ...
```

즉 구조는 다음과 같다.

```text
.venv Python
  └── subprocess로 radioconda Python 호출
        ├── RTL-SDR 수신 실행
        └── HackRF 송신 실행
```

---

## 8. RadioConda 직접 활성화

장비 인식 문제를 확인할 때는 RadioConda를 직접 켤 수 있다.

PowerShell 또는 cmd에서:

```powershell
cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda
```

주의:

```text
RadioConda는 SDR 제어용이다.
AI 학습은 RadioConda가 아니라 D:\ai_projects\SDR\.venv에서 실행한다.
```

---

## 9. RTL-SDR 수신 환경

수신 장비:

```text
RTL-SDR Blog V4
```

수신 방식:

```text
SoapySDR를 통해 RTL-SDR 장치를 연다.
수신 IQ는 complex64 형태로 저장한다.
```

정상 로그 예시:

```text
Found Rafael Micro R828D tuner
RTL-SDR Blog V4 Detected
[INFO] Opening Generic RTL2832U OEM :: 00000001...
[INFO] Using format CF32.
Captured 12000000 complex64 samples to ...
```

오류 로그 예시:

```text
Could not open RTL-SDR via SoapySDR: No RTL-SDR devices found!
```

주요 원인:

```text
RTL-SDR USB 연결 안 됨
다른 프로그램이 RTL-SDR을 사용 중
USB 포트 문제
드라이버 문제
RadioConda/SoapySDR 환경 문제
```

대응:

```text
1. RTL-SDR USB 재연결
2. SDR# / SDR++ / CubicSDR 등 수신 프로그램 종료
3. PowerShell 새로 열기
4. USB 포트 변경
5. PC 재부팅
```

---

## 10. HackRF One 송신 환경

송신 장비:

```text
HackRF One
```

송신 방식:

```text
Python에서 baseband IQ를 생성한다.
SoapySDR를 통해 HackRF One을 연다.
HackRF가 IQ를 RF로 송신한다.
```

정상 로그 예시:

```text
[INFO] Opening HackRF One #0 ...
TX Oshea2018 modulation=BPSK offset_hz=500000.0 amp=0.0 vga=30.0
```

중요:

```text
tx_amp_gain=0은 "송신 안 함"이라는 뜻이 아니다.
HackRF에는 별도 RF amp enable과 VGA gain이 있다.
현재 실험에서는 tx_amp_gain=0으로 두고 tx_vga_gain으로 송신 세기를 조절한다.
```

송신 세기 조절:

```text
tx_vga_gain: 주요 송신 세기 조절 값
tx_amp_gain: HackRF RF amp 관련 값, 보통 0 유지
```

오류 가능성:

```text
HackRF USB 연결 안 됨
다른 프로그램이 HackRF를 사용 중
송신 gain이 너무 낮아 수신 신호가 noise floor와 구분되지 않음
송신 gain이 너무 높아 RTL-SDR에서 clipping 발생
```

---

## 11. AGC 설정

RTL-SDR의 AGC는 꺼두는 것이 좋다.

이유:

```text
BASK는 amplitude 변화가 핵심 특징이다.
AGC가 켜져 있으면 수신기가 amplitude 변화를 자동 보정해 BASK 특징이 약해질 수 있다.
```

권장:

```text
agc: false
rx_gain: manual
```

config 예시:

```yaml
sdr:
  agc: false
  rx_gain: 30
```

---

## 12. 기본 RF 파라미터 의미

자주 쓰는 설정:

```yaml
sdr:
  center_freq: 433920000
  tx_sample_rate: 2400000
  rx_sample_rate: 2400000
  symbol_rate: 5000
  tx_vga_gain: 30
  tx_amp_gain: 0
  rx_gain: 30
  baseband_offset_hz: 500000
```

의미:

| 항목 | 의미 |
| --- | --- |
| `center_freq` | RTL-SDR/HackRF 중심 주파수 |
| `tx_sample_rate` | HackRF 송신 IQ sample rate |
| `rx_sample_rate` | RTL-SDR 수신 IQ sample rate |
| `symbol_rate` | 변조 bit/symbol 속도 |
| `tx_vga_gain` | HackRF 송신 gain |
| `tx_amp_gain` | HackRF RF amp 관련 gain |
| `rx_gain` | RTL-SDR 수신 gain |
| `baseband_offset_hz` | DC spike를 피하기 위한 송신 offset |

---

## 13. Baseband Offset이 필요한 이유

RTL-SDR 같은 direct conversion 수신기는 중심 주파수 근처에 DC spike가 생길 수 있다.

따라서 신호를 정확히 center에 두지 않고, 중심에서 떨어진 위치로 송신한다.

예:

```text
center_freq = 433.92 MHz
baseband_offset_hz = 500 kHz
실제 신호 중심 = 434.42 MHz 근처
```

이후 수신 데이터 처리에서 offset 위치의 신호를 다시 baseband로 내린다.

이 과정을 보통 다음처럼 부른다.

```text
channelize
frequency shift to baseband
downsample
```

---

## 14. Channelize / Downsample 개념

RTL-SDR은 넓은 대역을 수신한다.

예:

```text
2.4 MS/s로 수신
```

그 안에서 실제 관심 신호는 offset 근처의 좁은 대역에 있다.

처리 과정:

```text
1. baseband_offset_hz 위치의 신호를 DC 근처로 이동
2. 필요한 대역만 low-pass filter
3. target sample rate로 downsample
```

예:

```text
raw sample rate: 2.4 MS/s
channel bandwidth: 100 kHz
target sample rate: 160 kS/s
```

이 처리는 수신 측에서 알 수 있는 주파수/대역 처리이며, payload나 정답 label을 쓰지 않는다.

---

## 15. Gain 설정 시 주의점

Gain이 너무 낮으면:

```text
TX 신호가 noise floor와 구분되지 않는다.
quality_pass=false가 발생한다.
모델이 실제 변조가 아니라 잡음을 학습한다.
```

Gain이 너무 높으면:

```text
RTL-SDR ADC clipping 발생
I/Q 값이 최대/최소 근처에 붙음
constellation이 찌그러짐
세 변조의 특징이 망가짐
```

확인 지표:

```text
tx/noise RMS ratio
estimated SNR
clipping rate
FFT peak prominence
```

권장 판단:

```text
tx/noise RMS ratio가 충분히 커야 한다.
clipping rate는 낮아야 한다.
BASK/BFSK/BPSK 모두 비슷한 pass rate가 나와야 한다.
BFSK만 잘 잡히고 BASK/BPSK가 실패하면 좋은 조건이 아니다.
```

---

## 16. Random Payload 원칙

변조 분류에서는 payload pattern이 class shortcut이 되면 안 된다.

나쁜 예:

```text
BASK = 항상 101010...
BFSK = 항상 111000...
BPSK = 항상 000111...
```

이 경우 모델은 변조 특성이 아니라 bit pattern을 외울 수 있다.

권장:

```text
BASK/BFSK/BPSK 모두 capture마다 random payload bits 사용
payload seed는 metadata에만 기록
payload 자체는 모델 입력으로 사용하지 않음
```

이 프로젝트의 수집 스크립트는 다음 정보를 metadata에 남긴다.

```text
payload_seed
random_payload_bits: true
```

---

## 17. 수집 로그에서 봐야 하는 것

수신 시작 로그:

```text
RX: cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat ...
```

송신 시작 로그:

```text
TX: cmd /c call C:\Users\qus70\radioconda\Scripts\activate.bat ...
```

RTL-SDR 인식:

```text
RTL-SDR Blog V4 Detected
```

HackRF 인식:

```text
Opening HackRF One
```

저장 완료:

```text
Captured ... complex64 samples to ...
```

품질 확인:

```text
QUALITY: {
  "tx_to_noise_rms_ratio": ...,
  "estimated_snr_db": ...,
  "clipping_rate": ...,
  "quality_pass": true
}
```

---

## 18. 장비 점검용 최소 실행 예시

아래 명령은 실제 수집/송신을 수행하므로 장비 연결 후 실행한다.

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode

$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG_FIXED = "..\config\config.oshea2018.fixed.yaml"

& $PY -m src.experiment.run_oshea2018_capture_session `
  --config $CFG_FIXED `
  --session-id session_001 `
  --output-root ..\data\hardware_check `
  --captures-per-class 1 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --rx-gain 30 `
  --baseband-offset-hz 500000 `
  --max-retries 1
```

이 명령은 다음을 확인하는 데 사용한다.

```text
RTL-SDR 수신 가능 여부
HackRF 송신 가능 여부
RadioConda에서 SoapySDR 장비 접근 가능 여부
IQ 파일 저장 가능 여부
품질 지표 출력 여부
```

---

## 19. 환경 문제 해결 체크리스트

### Python / AI 쪽

```text
(.venv)가 활성화되어 있는가?
$PY가 D:\ai_projects\SDR\.venv\Scripts\python.exe를 가리키는가?
PYTHONPATH가 sourcecode로 잡혀 있는가?
torch.cuda.is_available()가 True인가?
nvidia-smi에 python.exe가 보이는가?
```

### SDR 쪽

```text
HackRF One이 USB에 연결되어 있는가?
RTL-SDR Blog V4가 USB에 연결되어 있는가?
다른 SDR 프로그램이 장비를 점유하고 있지 않은가?
radioconda 경로가 C:\Users\qus70\radioconda로 맞는가?
수집 로그에 RTL-SDR Blog V4 Detected가 보이는가?
송신 로그에 Opening HackRF One이 보이는가?
```

### RF/OTA 쪽

```text
안테나 간 거리가 약 1 m인가?
안테나가 서로 마주 보고 있는가?
TX gain이 너무 낮지 않은가?
RX gain이 너무 낮거나 높지 않은가?
clipping_rate가 높지 않은가?
quality_pass가 반복적으로 false가 아닌가?
```

---

## 20. 새 세션 시작 시 추천 순서

새 PowerShell:

```powershell
cd D:\ai_projects\SDR
.\.venv\Scripts\Activate.ps1

$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
& $PY -c "import torch; print(torch.cuda.is_available())"
```

Oshea2018 코드 환경:

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
```

SDR 장비 확인:

```powershell
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config ..\config\config.oshea2018.fixed.yaml `
  --session-id session_001 `
  --output-root ..\data\hardware_check `
  --captures-per-class 1 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --rx-gain 30 `
  --baseband-offset-hz 500000 `
  --max-retries 1
```

정상 기준:

```text
RTL-SDR Blog V4 Detected
Opening HackRF One
Captured ... complex64 samples
QUALITY ... quality_pass ...
```
