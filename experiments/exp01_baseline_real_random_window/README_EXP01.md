# 실험 1 사용 설명서: Real Random-Window Baseline

## 1. 실험 1 정의

실험 1은 SDR 변조 분류 프로젝트의 첫 번째 실제 장비 baseline이다. HackRF One으로
송신하고 RTL-SDR Blog V4로 수신한 real IQ 데이터를 사용해 BASK, BFSK, BPSK를
분류한다. 핵심 방식은 송신이 켜져 있는 구간에서 random window를 뽑아 train/val/test로
나눈 뒤 1D CNN을 학습하는 것이다.

이 폴더는 실험 1 보존본이다. 실험 2를 진행할 때 이 폴더를 덮어쓰지 않는다. 실험 2가
실패하거나 데이터가 꼬이면 이 폴더를 기준으로 다시 복구한다.

기준 결과:

- 데이터셋: 실제 SDR capture 기반 random-window dataset
- 클래스 순서: `BASK`, `BFSK`, `BPSK`
- 최종 test accuracy: 약 `0.8193`
- best validation accuracy: 약 `0.8207`
- checkpoint: `experiments/exp01_baseline_real_random_window/results/checkpoints/best.pt`

## 2. 장비 및 신호 조건

- 송신 장비: HackRF One
- 수신 장비: RTL-SDR Blog V4
- center frequency: `433 MHz`
- TX sample rate: `8 MS/s`
- RX sample rate: `2.4 MS/s`
- symbol rate: `5 ksps`
- TX VGA gain: `30`
- TX duration: `5 s`
- capture duration: `10 s`
- payload:
  - `BASK` -> `A`
  - `BFSK` -> `F`
  - `BPSK` -> `P`

권장 RF 연결:

- HackRF One TX와 RTL-SDR RX는 감쇠기를 넣고 유선으로 연결한다.
- 감쇠량은 `30-60 dB` 범위에서 시작한다.
- TX gain을 무작정 올리지 않는다. 실험 1에서는 `--tx-vga-gain 30`이 유효했고,
  `40`은 더 좋아지지 않았다.

## 3. 보존 폴더 구조

중요 산출물은 아래에 보존되어 있다.

```text
experiments/exp01_baseline_real_random_window/
  README_EXP01.md
  manifest_exp01.json
  config/
    config.exp01.yaml
    requirements.txt
  code_snapshot/
    src/
    tests/
  data/
    raw_iq/
      bask_a.bin
      bfsk_f.bin
      bpsk_p.bin
    metadata/
    processed/
      train/
      val/
      test/
      manifest_real_sequence.json
  docs/
    README_ML_DL_ENV.md
  results/
    checkpoints/best.pt
    logs/eval_metrics.json
    logs/train_log.json
    confusion_matrices/confusion_matrix.png
    real_capture_analysis/
    experiment_report1.md
```

`data/real/`과 `results/`는 계속 실험용 작업 경로로 써도 된다. 이 실험 폴더는 기준선
보존용 복사본이다.

`code_snapshot/`에는 실험 1을 고정한 시점의 `src/`, `tests/` 코드가 들어 있다. 이후
코드 변경으로 실험 2가 깨질 경우 참고용으로 사용한다.

`manifest_exp01.json`에는 보존 파일 목록, 크기, SHA256 hash, 기준 metric, 재현 명령이
들어 있다.

## 4. 사전 환경 확인

모든 명령은 프로젝트 루트에서 실행한다.

```powershell
cd D:\ai_projects\SDR
```

Python 버전 확인:

```powershell
.\.venv\Scripts\python.exe --version
```

실험 1 기준 환경:

```text
Python 3.11.9
```

PyTorch/CUDA 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

실험 1 기준 출력:

```text
2.7.1+cu118
11.8
True
NVIDIA GeForce RTX 4060
```

SDR 장비 확인:

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

정상 기준:

- HackRF One이 `driver=hackrf`로 열린다.
- RTL-SDR Blog V4가 `driver=rtlsdr`로 열린다.
- 두 장비가 SoapySDR device 목록에 표시된다.

`SoapyVOLKConverters: no VOLK config file found` 경고는 실험 1 진행에 치명적이지 않다.

## 5. 실험 1 전체 재현 절차

아래 절차는 현재 작업 경로인 `data/real/`과 `results/`를 사용해 실험 1을 다시 수행하는
명령이다.

### 5.1 real IQ capture

```powershell
cd D:\ai_projects\SDR

.\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --tx-vga-gain 30 --tx-seconds 5 --capture-seconds 10
```

이 명령의 역할:

- `BASK/A`, `BFSK/F`, `BPSK/P` 순서로 자동 송수신한다.
- RX를 먼저 켠 뒤 TX를 시작한다.
- raw IQ 파일을 아래 경로에 저장한다.
  - `data\real\raw_iq\bask_a.bin`
  - `data\real\raw_iq\bfsk_f.bin`
  - `data\real\raw_iq\bpsk_p.bin`

기대 파일 크기:

- `10 s` capture, `2.4 MS/s`, complex64 기준
- 각 파일 약 `192,000,000` bytes

### 5.2 이전 processed dataset 제거

```powershell
Remove-Item -Recurse -Force data\real\processed
```

이 단계는 이전 실험의 processed `.npz` 파일이 새 실험 데이터와 섞이지 않게 지운다.

### 5.3 random-window import

```powershell
.\.venv\Scripts\python.exe -m src.dataset.import_real_sequence --windows-per-class 1500 --active-start-seconds 1.1 --active-duration-seconds 4.5
```

이 명령의 역할:

- 세 raw IQ 파일을 읽는다.
- 송신 활성 구간을 `1.1 s`부터 `4.5 s` 길이로 본다.
- 각 modulation마다 random window `1500`개를 만든다.
- train/val/test split을 `data\real\processed` 아래에 만든다.

기대 split 개수:

```text
train: 3150
val:    675
test:   675
```

확인 명령:

```powershell
Get-ChildItem data\real\processed -Directory | ForEach-Object { [pscustomobject]@{Split=$_.Name; Count=(Get-ChildItem $_.FullName -Filter *.npz -File).Count} } | Format-Table -AutoSize
```

### 5.4 1D CNN 학습

```powershell
.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config config.yaml --preset train
```

이 명령의 역할:

- `config.yaml`의 `dataset.root`를 기준으로 real processed dataset을 읽는다.
- 1D CNN을 학습한다.
- best checkpoint를 `results\checkpoints\best.pt`에 저장한다.
- 학습 로그를 `results\logs\train_log.json`에 저장한다.

실험 1 기준:

- best validation accuracy는 약 `0.8207`
- preserved run의 best epoch는 `28`

### 5.5 평가

```powershell
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt --data-root data/real/processed
```

주의:

- 사용자가 중간에 적었던 `--data-root data/re`는 오타다.
- 올바른 경로는 `--data-root data/real/processed`다.

실험 1 기준 출력:

```text
accuracy=0.8193
```

보존된 기준 metric:

- Accuracy: `0.8192592592592592`
- Macro F1: `0.8189116058417459`

보존된 confusion matrix:

```text
          Pred BASK  Pred BFSK  Pred BPSK
True BASK       195         12         18
True BFSK        36        167         22
True BPSK        23         11        191
```

## 6. 보존된 실험 1 복사본 평가

현재 작업 경로의 `data/real/processed` 또는 `results/checkpoints/best.pt`가 바뀌었더라도
아래 명령으로 실험 1 보존본만 평가할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint experiments/exp01_baseline_real_random_window/results/checkpoints/best.pt --data-root experiments/exp01_baseline_real_random_window/data/processed
```

기대 출력:

```text
accuracy=0.8193
```

## 7. 보존된 실험 1 데이터로 재학습

작업 경로가 바뀌었더라도 실험 1 processed data로 다시 학습하려면 아래 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config experiments/exp01_baseline_real_random_window/config/config.exp01.yaml --preset train
```

현재 학습 스크립트는 checkpoint를 기본 `results/` 아래에 쓴다. 실험 1 보존 checkpoint를
유지하려면 새 checkpoint는 실험 2 또는 별도 폴더로 복사해 관리한다.

## 8. 실험 1을 작업 경로로 복구

이 절차는 active workspace를 실험 1 상태로 되돌리고 싶을 때만 사용한다.

```powershell
Remove-Item -Recurse -Force data\real\processed
New-Item -ItemType Directory -Force data\real\processed | Out-Null
Copy-Item -Recurse -Force experiments\exp01_baseline_real_random_window\data\processed\* data\real\processed\

Copy-Item -Force experiments\exp01_baseline_real_random_window\data\raw_iq\bask_a.bin data\real\raw_iq\bask_a.bin
Copy-Item -Force experiments\exp01_baseline_real_random_window\data\raw_iq\bfsk_f.bin data\real\raw_iq\bfsk_f.bin
Copy-Item -Force experiments\exp01_baseline_real_random_window\data\raw_iq\bpsk_p.bin data\real\raw_iq\bpsk_p.bin

Copy-Item -Force experiments\exp01_baseline_real_random_window\results\checkpoints\best.pt results\checkpoints\best.pt
```

복구 후 검증:

```powershell
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt --data-root data/real/processed
```

## 9. 시행착오 정리

실험 1에서 확인한 중요한 시행착오는 다음과 같다.

- simulation data로만 학습한 모델은 simulation test에서는 잘 동작했지만 real capture에서는
  accuracy가 약 `0.3333`에 머물렀다.
- 수동으로 RX/TX를 따로 실행하면 송수신 타이밍이 흔들려 데이터 품질과 재현성이 떨어졌다.
- `run_sdr_capture_sequence`로 RX를 먼저 시작하고 TX를 뒤따라 실행하는 자동화 방식으로
  전환했다.
- TX gain이 낮을 때는 신호가 약했다.
- `--tx-vga-gain 30`에서 성능이 개선되었다.
- `--tx-vga-gain 40`은 더 좋은 결과를 만들지 못했고, 오히려 왜곡 가능성이 있었다.
- 시간 구간을 train/val/test로 나누는 방식은 실패했다. train loss가 거의 0에 가까워져도
  val accuracy가 `0.3333` 수준에 머물렀다.
- 최종적으로 송신 활성 구간에서 random window split을 사용하는 방식으로 바꿨고,
  real SDR data 기준 accuracy 약 `0.8193`을 얻었다.

## 10. 실험 2와의 분리 규칙

실험 2는 아래 폴더만 사용한다.

```text
experiments/exp02_session_generalization/
```

실험 2의 raw IQ, processed dataset, checkpoint, log, report는 모두 실험 2 폴더 아래에
둔다. 아래 실험 1 폴더는 baseline으로 유지한다.

```text
experiments/exp01_baseline_real_random_window/
```

실험 2가 실패해도 실험 1 폴더를 기준으로 다시 시작할 수 있어야 한다.
