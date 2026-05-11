# SDR Modulation Classification Project

HackRF One으로 송신하고 RTL-SDR Blog V4로 수신한 IQ 신호를 이용해
`BASK`, `BFSK`, `BPSK` 변조 방식을 분류하는 PyTorch 1D CNN 프로젝트입니다.

현재 기준 baseline은 **실험 1: HackRF One TX + RTL-SDR V4 RX real random-window baseline**입니다.

실험별 계획, 코드 snapshot, 데이터 위치, 결과 분석은 `experiments/` 아래에서 분리 관리합니다. 새 실험을 시작할 때는 `experiments/_template/`를 복사해서 `expNN_short_name` 폴더를 만들고, root `data/`와 root `results/`는 임시 작업 공간으로만 사용합니다.

실험 1의 목적은 다음과 같습니다.

- 시뮬레이션 IQ 데이터 생성부터 CNN 학습까지 전체 파이프라인을 검증한다.
- 실제 SDR 장비로 BASK/BFSK/BPSK 신호를 송수신한다.
- 수신한 real IQ capture에서 random window dataset을 만든다.
- 1D CNN이 실제 수신 IQ에서 변조 방식을 분류할 수 있는지 baseline을 만든다.

실험 1 기준 결과:

```text
Accuracy: 약 0.8193
Macro F1: 약 0.8189
```

> 주의: GitHub에는 raw IQ `.bin`, processed `.npz`, checkpoint `.pt` 같은 대용량 산출물을 올리지 않습니다.
> 팀원이 pull 받은 뒤 실험 1을 재현하려면 실제 HackRF One + RTL-SDR V4로 다시 capture하거나, 별도로 공유받은 대용량 데이터를 `data/real/`에 넣어야 합니다.

---

## 1. 프로젝트 구조

```text
SDR/
  config.yaml
  requirements.txt
  README.md
  experiments/
    README.md
    _template/
    exp1/
      README.md
      PLAN.md
      RUNBOOK.md
      ANALYSIS.md
      config/
      data/
      results/
    exp2/
      README.md
      PLAN.md
      RUNBOOK.md
      ANALYSIS.md
      config/
      data/
      results/
```

대용량 파일은 `.gitignore`로 제외되어 있습니다.

```text
data/real/raw_iq/
data/real/processed/
data/sim/
results/checkpoints/
experiments/*/data/raw_iq/
experiments/*/data/processed/
*.bin
*.npz
*.pt
```

---

## 2. GitHub에서 받은 뒤 환경 구성

Windows PowerShell 기준입니다.

```powershell
git clone https://github.com/Jistan7011/SDR-Project-with-DL.git
cd SDR-Project-with-DL
```

Python 3.11 가상환경을 만듭니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

CUDA 11.8 PyTorch를 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

나머지 패키지를 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

환경 확인:

```powershell
.\.venv\Scripts\python.exe --version

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

.\.venv\Scripts\python.exe -m pytest -q
```

기준 환경:

```text
Python 3.11.x
PyTorch CUDA 사용 가능하면 cuda available: True
CPU만 있어도 smoke test는 가능
```

---

## 3. SDR 장비 환경 구성

실제 실험 1을 수행하려면 다음 장비와 Windows SDR 환경이 필요합니다.

- HackRF One
- RTL-SDR Blog V4
- 안테나 기반 OTA 송수신 환경
- Radioconda
- SoapySDR
- Zadig WinUSB driver

정정: 실험 1과 실험 2는 유선 RF 연결이 아니라 HackRF One과 RTL-SDR V4에 각각
안테나를 연결한 OTA 방식으로 진행했다. 두 SDR 안테나는 약 `10 cm` 거리에서 옆에
나란히 세웠다.

Radioconda 위치는 기존 실험 기준으로 다음을 사용했습니다.

```text
C:\Users\qus70\radioconda
```

팀원 PC에서 Radioconda 설치 위치가 다르면 명령의 경로를 본인 환경에 맞게 바꾸면 됩니다.

SDR 장비 인식 확인:

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

정상이라면 다음이 확인되어야 합니다.

```text
driver=hackrf open ok
driver=rtlsdr open ok
HackRF One 표시
RTL-SDR Blog V4 표시
```

Windows에서는 `SoapyVOLKConverters: no VOLK config file found` 같은 경고가 나올 수 있습니다. 실험 진행 자체에는 치명적이지 않습니다.

---

## 4. 먼저 시뮬레이션으로 파이프라인 확인

실제 장비가 없어도 아래 smoke test는 가능합니다.

```powershell
.\.venv\Scripts\python.exe -m src.dataset.generate_sim_dataset --preset smoke

.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config config.yaml --preset smoke

.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt

.\.venv\Scripts\python.exe -m src.app.decode_iq --input data/sim/test/sample_000000.npz --checkpoint results/checkpoints/best.pt
```

시뮬레이션 데이터 생성 과정:

```text
payload 문자 선택
→ preamble + sync word + payload bits + CRC8 frame 생성
→ BASK/BFSK/BPSK baseband IQ 변조
→ AWGN 노이즈 추가
→ IQ 정규화
→ [2, 1024] CNN 입력 생성
```

시뮬레이션 채널은 현재 단순 AWGN 중심입니다. 실제 SDR의 주파수 offset, gain 특성, IQ imbalance, 외부 간섭까지 정교하게 반영한 채널은 아닙니다.

---

## 5. 실험 1: 실제 SDR capture 진행

실험 1 기준 조건:

```text
center_freq: 433 MHz
tx_sample_rate: 8 MS/s
rx_sample_rate: 2.4 MS/s
symbol_rate: 5 ksps
tx_vga_gain: 30
tx_seconds: 5
capture_seconds: 10
```

송신 payload는 실험 1에서 고정했습니다.

```text
BASK → "A"
BFSK → "F"
BPSK → "P"
```

각 payload는 1 byte 문자이고, frame은 다음 구조입니다.

```text
preamble 16 bits
sync word 8 bits
payload 8 bits
CRC8 8 bits
----------------
total 40 bits
```

실제 capture 실행:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --tx-vga-gain 30 --tx-seconds 5 --capture-seconds 10
```

이 명령은 다음을 자동으로 수행합니다.

```text
1. RTL-SDR RX 먼저 시작
2. 약 1초 후 HackRF TX 시작
3. BASK/A capture 저장
4. BFSK/F capture 저장
5. BPSK/P capture 저장
```

생성되는 raw IQ 파일:

```text
data/real/raw_iq/bask_a.bin
data/real/raw_iq/bfsk_f.bin
data/real/raw_iq/bpsk_p.bin
```

각 파일은 `10초`, `2.4 MS/s`, `complex64` 기준 약 `192 MB`입니다.

---

## 6. 실험 1: real IQ를 CNN 학습 데이터로 변환

기존 processed dataset을 제거합니다.

```powershell
Remove-Item -Recurse -Force data\real\processed
```

수신 IQ에서 random window dataset을 만듭니다.

```powershell
.\.venv\Scripts\python.exe -m src.dataset.import_real_sequence --windows-per-class 1500 --active-start-seconds 1.1 --active-duration-seconds 4.5
```

의미:

```text
각 modulation마다 1500개 window 생성
송신 활성 구간은 capture 시작 후 1.1초부터
4.5초 길이 구간에서 random window 추출
각 window 길이 = 1024 complex samples
```

예상 split 개수:

```text
train: 3150
val:    675
test:   675
```

확인 명령:

```powershell
Get-ChildItem data\real\processed -Directory | ForEach-Object { [pscustomobject]@{Split=$_.Name; Count=(Get-ChildItem $_.FullName -Filter *.npz -File).Count} } | Format-Table -AutoSize
```

---

## 7. 1D CNN 입력 형태

수신 IQ 하나는 complex signal입니다.

```text
IQ sample = I + jQ
```

CNN에는 complex 값을 직접 넣지 않고, 실수부와 허수부를 나눠 2채널로 넣습니다.

sample 하나의 shape:

```text
[2, 1024]
```

batch 입력 shape:

```text
[batch, 2, 1024]
```

예를 들어 batch size가 64이면:

```text
[64, 2, 1024]
```

label 매칭:

```text
bask_a.bin에서 나온 window → BASK
bfsk_f.bin에서 나온 window → BFSK
bpsk_p.bin에서 나온 window → BPSK
```

즉, 모델은 payload 문자를 맞히는 것이 아니라 수신 IQ window의 파형 특징을 보고 변조 방식을 분류합니다.

class index:

```text
0 → BASK
1 → BFSK
2 → BPSK
```

---

## 8. 실험 1: CNN 학습 및 평가

학습:

```powershell
.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config config.yaml --preset train
```

평가:

```powershell
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt --data-root data/real/processed
```

실험 1 기준 결과:

```text
accuracy=0.8193
```

평가 결과는 다음 위치에 저장됩니다.

```text
results/logs/eval_metrics.json
results/confusion_matrices/confusion_matrix.png
```

checkpoint는 다음 위치에 저장됩니다.

```text
results/checkpoints/best.pt
```

단, checkpoint는 `.gitignore`로 제외되므로 GitHub에는 올라가지 않습니다.

---

## 9. 실험 1 상세 설명서

더 자세한 실험 1 사용 설명서는 아래 파일에 있습니다.

```text
experiments/exp1/README.md
```

이 파일에는 다음이 정리되어 있습니다.

- 실험 1 목적
- 장비 조건
- capture 명령
- random-window import 명령
- 학습/평가 명령
- 실험 1 시행착오
- 실험 2와 분리 관리하는 방법

GitHub에는 실험 1의 raw IQ와 processed dataset이 포함되어 있지 않습니다.
대신 실험 절차와 metric, 로그, 보고서, 코드 snapshot은 포함되어 있습니다.

---

## 10. 현재까지의 시행착오 요약

시뮬레이션만으로 학습한 모델은 시뮬레이션 test에서는 잘 동작했지만, 실제 SDR capture에서는 accuracy가 약 `0.3333` 수준이었습니다. 3-class 문제에서 `0.3333`은 거의 찍는 수준입니다.

그래서 실제 HackRF/RTL-SDR로 capture한 real IQ를 직접 학습 데이터로 사용했습니다.

중요한 시행착오:

```text
1. TX gain이 낮으면 수신 신호가 약함
2. tx_vga_gain 30에서 baseline 성능 개선
3. tx_vga_gain 40은 더 좋아지지 않음
4. 시간 구간을 train/val/test로 나누면 val accuracy가 0.3333 근처에 머묾
5. 송신 활성 구간에서 random window를 뽑는 방식으로 바꾸자 accuracy 약 0.8193 달성
```

다만 실험 1은 아직 완전한 일반화 성능을 증명한 것은 아닙니다. 같은 raw capture 안에서 random window를 나눴기 때문에 session 특성이 학습에 섞였을 수 있습니다.

---

## 11. 다음 단계: 실험 2

실험 2는 아래 폴더에서 별도로 진행합니다.

```text
experiments/exp2/
```

실험 2의 목표:

```text
1. capture session을 여러 번 반복
2. train session과 test session을 분리
3. payload를 class마다 고정하지 않고 다양화
4. gain, frequency offset, 시간 offset 변화에 대한 일반화 확인
5. simulation과 real SDR 사이의 차이를 줄이는 augmentation/channel model 추가
```

실험 1은 “실제 장비에서 변조 분류가 가능함을 확인한 baseline”이고, 실험 2는 “다른 세션에서도 일반화되는지 확인하는 단계”입니다.

