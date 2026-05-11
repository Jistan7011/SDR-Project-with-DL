# 실험 2 최종 보고서 및 재현 가이드

작성일: 2026-05-05

이 문서는 실험 2의 최종 보고서, 분석, 재현 가이드입니다. 실험 2에서 실제로 따른 단일 계획은 `PLAN.md`에 정리되어 있습니다. 이 보고서는 무엇을 했고, 진행 중 어떤 수정이 있었고, 어떤 결과를 얻었으며, 다른 팀원이 `experiments/exp2` 폴더만 보고 어떻게 실험을 재현할 수 있는지 설명합니다.

## 1. 목적

실험 2의 목적은 실제 SDR capture에서 얻은 IQ 데이터로 학습한 변조 분류 모델이 보지 못한 capture session에서도 일반화되는지 확인하는 것입니다.

분류 문제는 다음과 같습니다.

```text
입력:  RTL-SDR V4로 수신한 real IQ window
출력:  BASK, BFSK, BPSK 중 하나
모델:  1D CNN 계열, 최종 선택 모델은 ResNet1D
```

실험 1은 같은 capture 파일에서 random window를 뽑아 train/test를 구성했습니다. 이 방식은 비교적 높은 accuracy를 냈지만, train과 test가 같은 RF 조건을 공유할 수 있었습니다. 실험 2는 이 문제를 해결하기 위해 `session-held-out` 평가를 사용했습니다.

## 2. Session-Held-Out의 의미

session은 하나의 독립적인 RF capture 조건 묶음입니다. 같은 raw 파일에서 window만 다르게 뽑은 것은 새로운 session이 아닙니다.

실험 2의 최종 split은 다음과 같습니다.

```text
train: session_001 ~ session_009
val:   session_010 ~ session_012
test:  session_013 ~ session_015
```

test session의 window는 train 또는 validation에 절대 들어가지 않습니다. 이 조건은 모델이 특정 capture 조건을 외운 것이 아니라 변조 방식 자체의 특징을 학습했는지 확인하기 위한 장치입니다.

## 3. 장비 및 신호 설정

사용 장비와 기본 조건:

```text
TX: HackRF One
RX: RTL-SDR Blog V4
center frequency: 433 MHz
TX sample rate: 8 MS/s
RX sample rate: 2.4 MS/s
symbol rate: 5 ksps
processing 후 target sample rate: 160 kS/s
window size: 1024
RF 송수신 방식: 안테나 기반 OTA
TX/RX 안테나 거리: 약 10 cm
TX/RX 배치: 두 SDR 안테나를 옆에 나란히 세운 근거리 배치
```

중요 정정: 실험 2는 HackRF One과 RTL-SDR V4를 동축 케이블과 감쇠기로 직결한 실험이
아니다. 실험 1과 동일하게 두 장비 모두 안테나를 사용했고, 약 `10 cm` 거리에서 나란히
세운 OTA 근거리 조건으로 session을 수집했다. 따라서 실험 2의 `0.7594` ensemble
accuracy는 `10 cm OTA session-held-out` 조건의 결과로 해석해야 한다.

payload pool:

```text
["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
```

모든 modulation class가 같은 payload pool을 사용합니다. 이는 모델이 “payload A는 BASK” 같은 shortcut을 외우는 것을 막기 위한 핵심 수정입니다.

frame 구조:

```text
preamble:  1010101010101010   16 bits
sync word: 11001100           8 bits
payload:   ASCII 문자 1개      8 bits
CRC8:      8 bits
total:     40 bits
```

모델 입력 변환:

```text
raw IQ capture
-> baseband offset 기준 channelize
-> low-pass filter
-> 160 kS/s로 downsample
-> 1024-sample window 추출
-> [I, Q] 저장
-> model input shape: [batch, 2, 1024]
```

## 4. 폴더 구조

실험 2 명령은 기본적으로 아래 위치에서 실행합니다.

```powershell
cd D:\ai_projects\SDR\experiments\exp2\sourcecode
```

주요 경로:

```text
experiments/exp2/config/
  config.exp02.yaml
  config.exp02.aug.yaml
  config.exp02.ifreq.yaml

experiments/exp2/data/raw_iq/
  session_001/
  ...
  session_015/

experiments/exp2/data/processed/
  train/
  val/
  test/

experiments/exp2/results/
  exp02_15session/
  exp02_15session_aug/
  exp02_15session_vgg/
  exp02_15session_resnet/
  exp02_15session_resnet_ifreq/
  reports/
  figures/final_report/
```

## 5. 재현 명령어

### 5.1 Python 및 SDR 환경 확인

프로젝트 루트에서 실행합니다.

```powershell
cd D:\ai_projects\SDR

.\.venv\Scripts\python.exe --version

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Radioconda 환경에서 SDR 장비를 확인합니다.

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

그 다음 exp2 source 폴더로 이동합니다.

```powershell
cd D:\ai_projects\SDR\experiments\exp2\sourcecode
```

선택 사항으로 RTL-SDR sample rate gate를 실행할 수 있습니다.

```powershell
..\..\..\.venv\Scripts\python.exe -m src.sdr.rtl_rate_check --radioconda-root C:\Users\qus70\radioconda --sample-rate 2400000 --seconds 10
```

drop이 확인되면 이후 capture에서 RX sample rate를 2.048 MS/s로 낮춥니다.

### 5.2 Session Capture

각 session은 noise-only capture를 먼저 저장하고, 그 다음 BASK/BFSK/BPSK를 모든 payload에 대해 송수신합니다.

예시 명령:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp02_capture_session --session-id session_001 --baseband-offset-hz 500000 --tx-vga-gain 30 --rx-gain 30 --tx-seconds 5 --capture-seconds 10 --noise-seconds 3

..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp02_capture_session --session-id session_002 --baseband-offset-hz 500000 --tx-vga-gain 25 --rx-gain 30 --tx-seconds 5 --capture-seconds 10 --noise-seconds 3

..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp02_capture_session --session-id session_003 --baseband-offset-hz 500000 --tx-vga-gain 35 --rx-gain 30 --tx-seconds 5 --capture-seconds 10 --noise-seconds 3
```

완료된 실험 2는 총 15개 formal session을 사용했습니다. raw session 폴더가 이미 포함되어 있다면 capture 단계는 건너뛰고 dataset import부터 시작하면 됩니다.

### 5.3 Dataset Import

raw IQ에서 train/val/test processed dataset을 다시 생성합니다.

```powershell
if (Test-Path ..\data\processed) { Remove-Item -Recurse -Force ..\data\processed }

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp02_sessions --min-sessions 15
```

예상 결과:

```text
train: 21600 samples
val:    7200 samples
test:   7200 samples
total: 36000 samples
shape: [2, 1024]
```

### 5.4 테스트

```powershell
..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

예상 결과:

```text
9 passed
```

### 5.5 모델 학습 및 평가

CNN1D baseline:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.yaml --model-type cnn1d --output-root ..\results\exp02_15session
```

CNN1D + augmentation:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.aug.yaml --model-type cnn1d --output-root ..\results\exp02_15session_aug
```

VGG1D:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.yaml --model-type vgg1d --output-root ..\results\exp02_15session_vgg
```

공식 모델 ResNet1D:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.yaml --model-type resnet1d --output-root ..\results\exp02_15session_resnet
```

instantaneous-frequency ablation:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.ifreq.yaml --model-type resnet1d --output-root ..\results\exp02_15session_resnet_ifreq
```

### 5.6 오류 분석

공식 ResNet1D 오류 분석:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_resnet_errors --config ..\config\config.exp02.yaml --data-root ..\data\processed --output-dir ..\results\reports
```

instantaneous-frequency 모델 오류 분석:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_resnet_errors --config ..\config\config.exp02.ifreq.yaml --data-root ..\data\processed --checkpoints ..\results\exp02_15session_resnet_ifreq\resnet1d_seed42\checkpoints\best.pt ..\results\exp02_15session_resnet_ifreq\resnet1d_seed43\checkpoints\best.pt ..\results\exp02_15session_resnet_ifreq\resnet1d_seed44\checkpoints\best.pt --output-dir ..\results\reports\ifreq
```

### 5.7 최종 그림 재생성

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.make_final_report_figures
```

출력 위치:

```text
..\results\figures\final_report\
```

## 6. 실험 계획과 결정 기준

원래 계획:

```text
1. 모든 modulation class가 같은 payload pool을 사용한다.
2. 최소 12 sessions, 목표 15 sessions를 수집한다.
3. train/val/test는 session 단위로만 나눈다.
4. overall accuracy뿐 아니라 condition-wise accuracy도 평가한다.
5. CNN1D부터 시작한다.
6. recall이 불안정하면 augmentation, VGG1D, ResNet1D 순서로 확장한다.
7. targeted preprocessing ablation을 한 번 수행한 뒤 개선이 작으면 실험 2를 종료한다.
```

성공 기준:

```text
session-held-out test accuracy >= 0.75
all class recall >= 0.65
3-seed evaluation 완료
condition-wise analysis 완료
명백한 class collapse가 없을 것
```

개선 중단 기준:

```text
preprocessing ablation을 한 번 수행한 뒤
accuracy가 약 +2 percentage points 이상 개선되지 않고,
공식 모델이 이미 기준을 통과했다면,
실험 2 개선을 중단하고 최종 보고서로 정리한다.
```

## 7. 시행착오

### 7.1 실험 1만으로는 부족했던 이유

실험 1은 real random window 기준 약 `0.8193` accuracy를 얻었습니다. 그러나 같은 capture 파일에서 나온 window가 train과 test에 함께 들어갈 수 있었습니다. 따라서 이 점수는 baseline으로는 유용하지만 session 일반화 성능을 증명하기에는 부족했습니다.

### 7.2 Session 수 확대

초기 계획보다 session 수를 늘려 최종적으로 15 sessions를 사용했습니다.

```text
train: 9 sessions
val:   3 sessions
test:  3 sessions
```

### 7.3 Offset과 Gain의 영향

100 kHz offset은 DC spike에 가까울 수 있어 primary 조건에서 제외했습니다. 주된 offset은 250 kHz와 500 kHz였습니다.

최종 condition analysis:

```text
250 kHz offset accuracy: 0.7667
500 kHz offset accuracy: 0.7558
RX gain 30 accuracy:     0.7648
RX gain 35 accuracy:     0.7488
```

### 7.4 모델 확장

CNN1D는 overall accuracy는 통과했지만 recall이 불안정했습니다. augmentation은 recall을 약간 개선했습니다. VGG1D는 뚜렷한 개선이 없었습니다. ResNet1D는 accuracy와 class recall 기준을 모두 통과했습니다.

### 7.5 Feature Ablation

주요 오류는 다음 형태였습니다.

```text
BFSK -> BASK
BPSK -> BASK
```

따라서 instantaneous-frequency 입력 channel을 추가한 ablation을 수행했습니다. 이 방식은 worst-class recall을 개선했지만 overall accuracy를 약간 낮췄습니다. 그래서 공식 모델로 채택하지 않고 분석 결과로만 남겼습니다.

## 8. 결과

![모델 비교](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/model_comparison.png)

| 모델 | Mean Accuracy | Mean Macro F1 | Worst Class Recall | 판단 |
|---|---:|---:|---:|---|
| CNN1D | 0.7583 | 0.7615 | 0.6313 | accuracy는 통과, recall 약함 |
| CNN1D + Augmentation | 0.7579 | 0.7605 | 0.6471 | recall 기준에 근접 |
| VGG1D | 0.7583 | 0.7618 | 0.6325 | 개선 없음 |
| ResNet1D | 0.7579 | 0.7588 | 0.6558 | 공식 모델 |
| ResNet1D + ifreq | 0.7553 | 0.7552 | 0.7038 | 균형은 개선, accuracy는 하락 |

공식 결과:

```text
official model: ResNet1D [I,Q]
mean accuracy: 0.7579
worst observed recall: 0.6558
majority-vote ensemble accuracy: 0.7594
```

## 9. 분석

ResNet1D majority-vote ensemble:

```text
test samples: 7200
ensemble accuracy: 0.7594
ensemble recall BASK: 0.9062
ensemble recall BFSK: 0.7183
ensemble recall BPSK: 0.6538
all-seed agreement fraction: 0.7239
all-seed agreement accuracy: 0.9210
seed-disagreement fraction: 0.2761
seed-disagreement accuracy: 0.3360
```

seed agreement는 confidence signal로 쓸 수 있습니다. 세 seed가 모두 같은 예측을 할 때 accuracy가 높았고, seed 간 예측이 갈릴 때는 어려운 sample일 가능성이 컸습니다.

대표 confusion matrix:

![ResNet confusion matrix](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/resnet_confusion_matrix_seed42.png)

session별 accuracy:

![Session accuracy](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/session_accuracy.png)

payload별 accuracy:

![Payload accuracy](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/payload_accuracy.png)

SNR bin별 accuracy:

![SNR bin accuracy](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/snr_bin_accuracy.png)

offset 및 RX gain별 accuracy:

![Offset accuracy](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/offset_accuracy.png)

![RX gain accuracy](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/rx_gain_accuracy.png)

주요 오류 흐름:

![Top error flows](D:/ai_projects/SDR/experiments/exp2/results/figures/final_report/top_error_flows.png)

## 10. 최종 결론

실험 2는 목적을 달성했습니다.

실제 SDR BASK/BFSK/BPSK 분류가 더 엄격한 session-held-out 조건에서도 일정 수준 일반화될 수 있음을 확인했습니다. 다만 margin이 크지는 않으므로 후속 실험이 필요합니다.

최종 결정:

```text
실험 2 공식 모델은 ResNet1D [I,Q]로 유지한다.
실험 2 개선 run은 더 진행하지 않는다.
feature fusion 또는 장기 일반화 실험은 실험 3으로 넘긴다.
```

## 11. 한계

- test session이 3개뿐입니다.
- 실험 2의 모든 실제 capture는 약 `10 cm` OTA 근거리 배치에서 수행되었습니다.
  `1 m` 이상 거리 변화에 대한 일반화는 아직 검증하지 않았습니다.
- 날짜가 크게 달라지는 long-term drift는 충분히 검증하지 못했습니다.
- real-time inference는 검증하지 않았습니다.
- estimated SNR만으로 난이도를 완전히 설명하지 못했습니다.
- `[I,Q,ifreq]`는 balance를 개선했지만 top-line accuracy를 개선하지는 못했습니다.

## 12. 실험 3 후보

실험 3으로 이어갈 만한 방향:

```text
1. 10 cm OTA에서 1 m OTA로 거리 변화 일반화 검증
2. [I,Q], magnitude, instantaneous frequency, PSD feature fusion
3. unseen-day 또는 unseen-position session
4. real-time capture to inference pipeline
5. seed/model agreement 기반 confidence-aware inference
```

## 13. 관련 파일

현재 exp2 루트의 문서 구조:

```text
README.md
PLAN.md
FINAL_REPORT_EXP02.md
```

세부 결과 파일:

```text
results/reports/resnet1d_15session_seed_sweep_summary.md
results/reports/resnet1d_error_analysis.md
results/reports/resnet1d_ifreq_ablation_summary.md
results/figures/final_report/
```
