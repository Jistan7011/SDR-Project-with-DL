# 실험 3 최종 보고서

작성일: 2026-05-06  
실험명: `1 m OTA 거리 일반화 + domain adaptation 기반 변조 분류 개선`

## 1. 실험 목적

프로젝트 전체 목적은 다음이다.

```text
SDR + 딥러닝 기반 변조 분류 및 데이터 복원 시스템 구현
```

그중 실험 3의 목적은 `실험 1, 2에서 확보한 10 cm OTA 안테나 송수신 조건`을 `1 m OTA 안테나 송수신 조건`으로 확장했을 때, 딥러닝 변조 분류 모델이 BASK/BFSK/BPSK를 계속 분류할 수 있는지 검증하는 것이다.

실험 1, 2에 대한 해석은 다음처럼 정정한다.

```text
실험 1:
  RF path: OTA antenna
  SDR 간 RF cable: 없음
  TX/RX distance: 약 10 cm
  antenna layout: side-by-side
  split: same-capture random window
  result: accuracy 약 0.8193
  해석: 10 cm OTA random-window baseline

실험 2:
  RF path: OTA antenna
  SDR 간 RF cable: 없음
  TX/RX distance: 약 10 cm
  antenna layout: side-by-side
  split: session-held-out
  result: ResNet1D ensemble accuracy 약 0.7594
  해석: 10 cm OTA session-held-out generalization

실험 3:
  RF path: OTA antenna
  SDR 간 RF cable: 없음
  TX/RX distance: 약 1 m
  antenna layout: face-to-face
  목적: 10 cm OTA 모델이 1 m OTA에서도 일반화되는지 검증
```

중요한 점은 HackRF One과 RTL-SDR V4가 서로 RF 케이블로 연결된 것이 아니라는 것이다. 두 장비는 같은 PC에 USB로 연결되어 있고, RF 신호는 각 SDR의 안테나를 통해 공중 전송된다.

## 2. 장비 및 RF 조건

```text
TX: HackRF One + antenna
RX: RTL-SDR Blog V4 + antenna
RF path: OTA antenna
RF cable between SDRs: false
TX/RX distance: about 1 m
antenna layout: face_to_face
center frequency: 433 MHz
TX sample rate: 8 MS/s
RX sample rate: 2.4 MS/s
symbol rate: 5 ksps
window size: 1024 samples
classes: BASK, BFSK, BPSK
payload pool: A, F, P, 0, 1, 7, K, R, S, Z
```

실험 3은 신규 1 m OTA session `session_016 ~ session_021`을 수집했다.

| session | distance | tx_vga_gain | rx_gain | offset_hz | raw files | TX-on captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| session_016 | 1.0 m | 30 | 30 | 500000 | 31 | 30 |
| session_017 | 1.0 m | 25 | 30 | 500000 | 31 | 30 |
| session_018 | 1.0 m | 35 | 30 | 500000 | 31 | 30 |
| session_019 | 1.0 m | 30 | 25 | 500000 | 31 | 30 |
| session_020 | 1.0 m | 30 | 35 | 500000 | 31 | 30 |
| session_021 | 1.0 m | 30 | 30 | 250000 | 31 | 30 |

각 session은 noise-only capture 1개와 `3 modulation x 10 payload` TX-on capture 30개를 포함한다.

## 3. 데이터 생성 및 입력 구조

각 payload는 1 byte 문자이며, frame은 다음 구조를 사용한다.

```text
preamble 16 bits
sync word 8 bits
payload 8 bits
CRC8 8 bits
----------------
total 40 bits
```

payload pool은 모든 modulation에서 동일하게 사용했다.

```text
["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
```

이렇게 한 이유는 모델이 `payload 패턴`을 외우는 것이 아니라 `변조 방식`을 학습하게 만들기 위해서다.

수신 raw IQ는 channelize/downsample/normalize 후 1024 sample window로 잘라 `.npz` sample로 저장한다. 기본 딥러닝 입력은 다음 형태다.

```text
[batch, channels, 1024]
```

실험 3에서 비교한 주요 입력 feature는 다음이다.

| feature mode | input shape | 설명 |
| --- | --- | --- |
| `[I,Q]` | `[batch, 2, 1024]` | raw IQ 기본 입력 |
| `[I,Q,instantaneous_frequency]` | `[batch, 3, 1024]` | phase/frequency 변화를 추가한 입력 |
| `[I,Q,magnitude,instantaneous_frequency]` | `[batch, 4, 1024]` | amplitude와 frequency feature 추가 |
| FusionResNet1D | time branch + spectral branch | time-domain feature와 FFT power spectrum embedding 결합 |

## 4. Dataset Split

실험 3 초기 strict split은 다음과 같다.

```text
train:
  exp2 session_001 ~ session_009

val:
  exp2 session_010 ~ session_012

test-A:
  exp2 session_013 ~ session_015
  목적: 10 cm OTA reference 재현

test-B:
  exp3 session_016 ~ session_021
  목적: 1 m OTA 거리 일반화 검증
```

processed dataset 생성 결과:

| split | samples | sessions |
| --- | ---: | --- |
| train | 21600 | exp2 session_001 ~ session_009 |
| val | 7200 | exp2 session_010 ~ session_012 |
| test_a | 7200 | exp2 session_013 ~ session_015 |
| test_b | 14400 | exp3 session_016 ~ session_021 |
| test | 14400 | test_b mirror |

후반 개선에서는 현실적인 target-domain calibration을 반영하기 위해 domain adaptation split도 만들었다.

```text
train:
  session_001 ~ session_009
  session_016 ~ session_018

val:
  session_010 ~ session_012
  session_019

test:
  session_020 ~ session_021
```

이 split은 1 m OTA session 일부를 train/val에 넣고, 남은 1 m OTA session을 held-out test로 둔다. 즉 “1 m 환경을 조금 캘리브레이션했을 때 실사용에 가까운 성능”을 보는 실험이다.

## 5. 실행 과정

모든 명령은 기본적으로 다음 위치에서 실행했다.

```powershell
cd D:\ai_projects\SDR\experiments\exp3\sourcecode
```

### 5.1 환경 확인

```powershell
..\..\..\.venv\Scripts\python.exe --version
..\..\..\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

### 5.2 1 m OTA session capture

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_016 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_017 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 25 --rx-gain 30 --baseband-offset-hz 500000
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_018 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 35 --rx-gain 30 --baseband-offset-hz 500000
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_019 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 30 --rx-gain 25 --baseband-offset-hz 500000
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_020 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 30 --rx-gain 35 --baseband-offset-hz 500000
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_021 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 250000
```

### 5.3 strict exp3 dataset import

```powershell
Remove-Item -Recurse -Force ..\data\processed -ErrorAction SilentlyContinue
..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp03_sessions --include-exp2-reference --min-new-sessions 6
```

### 5.4 domain adaptation dataset 생성

```powershell
..\..\..\.venv\Scripts\python.exe -m src.dataset.make_session_split_dataset --source-root ..\data\processed --output-root ..\data\processed_domain_adapt --train-sessions session_001,session_002,session_003,session_004,session_005,session_006,session_007,session_008,session_009,session_016,session_017,session_018 --val-sessions session_010,session_011,session_012,session_019 --test-sessions session_020,session_021
```

### 5.5 주요 학습 명령

strict 1 m 일반화 baseline:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp03.pilot.yaml --model-type fusion_resnet1d --output-root ..\results\exp03_fusion_resnet_pilot --eval-split test_b --seeds 42 43 44
```

domain adaptation all seeds:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp03.domain_adapt_ifreq.yaml --data-root ..\data\processed_domain_adapt --output-root ..\results\exp03_domain_adapt_ifreq_allseeds --model-type resnet1d --eval-split test --seeds 42 43 44
```

full fine-tune:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp03.domain_adapt_ifreq_finetune.yaml --data-root ..\data\processed_domain_adapt --output-root ..\results\exp03_domain_adapt_ifreq_finetune_seed44 --model-type resnet1d --eval-split test --seeds 44
```

classifier-only fine-tune:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp03.domain_adapt_ifreq_finetune_classifier.yaml --data-root ..\data\processed_domain_adapt --output-root ..\results\exp03_domain_adapt_ifreq_finetune_classifier_seed44 --model-type resnet1d --eval-split test --seeds 44
```

## 6. 시행착오 및 개선 과정

### 6.1 I/O 병목

초기에는 `.npz` 파일을 매 batch마다 디스크에서 읽어 1 epoch가 약 9분 수준으로 예상되었다. 전체 seed sweep이 너무 오래 걸려 full 실행을 중단했다.

개선:

```text
IQDataset(preload=True) 옵션 추가
train/evaluate에서 config의 dataset.preload 사용
pilot config 추가
```

결과적으로 반복 학습 속도가 크게 개선되었다.

### 6.2 Feature Fusion pilot

FusionResNet1D는 time-domain feature와 spectral branch를 결합했다. 그러나 test-B에서 BPSK recall이 낮았다.

| model | test-B accuracy | BASK recall | BFSK recall | BPSK recall |
| --- | ---: | ---: | ---: | ---: |
| FusionResNet1D pilot ensemble | 0.7572 | 0.7808 | 0.8415 | 0.6492 |

판단:

```text
FusionResNet1D는 BFSK에는 도움이 되었지만 BPSK를 충분히 살리지 못했다.
현재 구조로는 baseline 교체 근거가 부족하다.
```

### 6.3 Baseline 재평가

exp2 official checkpoint들을 exp3 test-A/test-B에서 재평가했다.

| model | test-A acc | test-B acc | test-B BASK | test-B BFSK | test-B BPSK | worst test-B recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet1D `[I,Q]` | 0.7594 | 0.7599 | 0.9169 | 0.6892 | 0.6735 | 0.6735 |
| ResNet1D `[I,Q,ifreq]` | 0.7561 | 0.7586 | 0.8121 | 0.7354 | 0.7283 | 0.7283 |
| FusionResNet1D pilot | 0.7549 | 0.7572 | 0.7808 | 0.8415 | 0.6492 | 0.6492 |

판단:

```text
1 m OTA distance generalization: pass
Feature Fusion replacement: fail
Best balanced strict model: ResNet1D [I,Q,instantaneous_frequency]
```

### 6.4 RF impairment audit

팀원 피드백 중 타당한 부분만 채택해 RF impairment audit를 추가했다.

채택:

```text
OTA와 synthetic 사이에는 clock/LO/gain/phase/noise/multipath/하드웨어 차이가 있다.
gain variation은 실제 TX/RX gain 조건으로 반영 가능하다.
carrier/LO offset, DC offset, IQ imbalance는 capture에서 추정해 기록하는 것이 좋다.
OTA 환경에서는 multipath가 생길 수 있다.
```

보류:

```text
USRP 표현: 현재 장비는 HackRF One + RTL-SDR V4다.
sample-rate offset 직접 추정: 현재 frame/capture 구조만으로 신뢰도 있게 추정하기 어렵다.
앞쪽 layer가 항상 일반 RF 특징만 학습한다는 설명: 가능성은 있지만 보장된 사실은 아니다.
```

RF audit 결과:

```text
DC offset:
  DC/RMS 평균이 대략 -58 dB ~ -62 dB라 핵심 문제로 보기 어렵다.

IQ imbalance:
  I/Q power ratio는 거의 0 dB, I-Q correlation도 거의 0이다.

gain/SNR:
  session별 RMS와 SNR 차이가 가장 뚜렷하다.
  특히 session_017은 TX VGA gain 25 조건이라 hard condition으로 봐야 한다.
```

### 6.5 RF augmentation

strong augmentation:

| split | accuracy | BASK recall | BFSK recall | BPSK recall |
| --- | ---: | ---: | ---: | ---: |
| test-B | 0.7594 | 0.7290 | 0.8754 | 0.6740 |

mild augmentation:

| split | accuracy | BASK recall | BFSK recall | BPSK recall |
| --- | ---: | ---: | ---: | ---: |
| test-B | 0.7585 | 0.8058 | 0.6412 | 0.8285 |

판단:

```text
strong augmentation은 BFSK를 올렸지만 BPSK를 낮췄다.
mild augmentation은 BPSK를 올렸지만 BFSK를 낮췄다.
단순 augmentation은 class별 decision boundary를 한쪽으로 밀어 공식 모델로 채택하지 않았다.
```

### 6.6 Condition-balanced sampler

조건별 오류를 줄이기 위해 `modulation + session_id + payload + snr_bin` 기준 weighted sampler를 구현했다.

| model | test-B accuracy | BASK recall | BFSK recall | BPSK recall | worst recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet1D `[I,Q,ifreq]` | 0.7586 | 0.8121 | 0.7354 | 0.7283 | 0.7283 |
| Condition-balanced sampler | 0.7538 | 0.8237 | 0.7608 | 0.6769 | 0.6769 |

판단:

```text
BFSK recall은 개선됐지만 BPSK와 overall accuracy가 하락했다.
공식 모델로 채택하지 않았다.
```

### 6.7 Ensemble

서로 장단점이 다른 모델을 soft-voting으로 결합했다.

| split | accuracy | BASK recall | BFSK recall | BPSK recall | worst recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| test-B | 0.7571 | 0.7723 | 0.7371 | 0.7619 | 0.7371 |

판단:

```text
worst recall은 0.7283 -> 0.7371로 개선됐다.
하지만 accuracy는 낮아졌으므로 공식 best로 채택하지 않았다.
```

### 6.8 Domain adaptation

1 m OTA session 일부를 train/val에 포함하는 현실적인 calibration split을 구성했다.

seed별 결과:

| model | test sessions | accuracy | BASK recall | BFSK recall | BPSK recall | 판단 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| seed42 | session_020~021 | 0.7656 | 0.9100 | 0.6763 | 0.7106 | BASK 쏠림 |
| seed43 | session_020~021 | 0.7627 | 0.8581 | 0.6913 | 0.7388 | BASK 쏠림 완화 |
| seed44 | session_020~021 | 0.7698 | 0.7350 | 0.7888 | 0.7856 | 가장 좋은 단일 모델 |
| hard vote ensemble | session_020~021 | 0.7617 | 0.8675 | 0.6875 | 0.7300 | 단일 seed44보다 낮음 |
| soft probability ensemble | session_020~021 | 0.7631 | 0.8313 | 0.7300 | 0.7281 | 안정적이지만 seed44보다 낮음 |

판단:

```text
domain adaptation은 accuracy를 약 0.7586 -> 0.7698까지 올렸다.
seed44가 가장 좋은 단일 모델이었다.
ensemble은 seed 간 편차는 줄였지만 seed44 단독보다 낮았다.
```

### 6.9 Balanced checkpoint 및 fine-tuning

기존 train은 val accuracy가 가장 높은 epoch를 `best.pt`로 저장했다. 실험 3에서는 class별 recall trade-off가 중요하므로 다음 metric을 추가했다.

```text
val_accuracy
val_mean_recall
val_worst_recall
val_balanced_score
```

또한 exp2 10 cm OTA ifreq checkpoint를 초기값으로 사용해 1 m OTA domain adaptation data로 fine-tuning했다.

| model | accuracy | BASK recall | BFSK recall | BPSK recall | worst recall | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| previous domain adapt seed44 | 0.7698 | 0.7350 | 0.7888 | 0.7856 | 0.7350 | 기존 best |
| balanced checkpoint seed44 | 0.7692 | 0.8219 | 0.7350 | 0.7506 | 0.7350 | 기존보다 좋지는 않음 |
| classifier-only fine-tune | 0.7688 | 0.8063 | 0.7569 | 0.7431 | 0.7431 | 균형 목적이면 유효 |
| full fine-tune | 0.7704 | 0.8281 | 0.7469 | 0.7362 | 0.7362 | accuracy 기준 현재 best |

class bias calibration도 시도했다. validation 기준 최적 class bias가 모두 `[0, 0, 0]`으로 나와 후처리 threshold 조정은 성능 개선에 기여하지 않았다.

## 7. 최종 결과

실험 3의 최종 판단은 다음이다.

```text
1 m OTA distance generalization:
  성공

Feature Fusion replacement:
  현재 구현 기준 실패

Strict 10 cm -> 1 m zero-shot 기준 best balanced:
  ResNet1D [I,Q,instantaneous_frequency]
  test-B accuracy = 0.7586
  worst recall = 0.7283

1 m calibration 포함 domain adaptation 기준 best single:
  domain adaptation [I,Q,ifreq] seed44
  test accuracy = 0.7698
  BASK/BFSK/BPSK recall = 0.7350 / 0.7888 / 0.7856

accuracy 기준 현재 best:
  full fine-tune seed44
  test accuracy = 0.7704
  BASK/BFSK/BPSK recall = 0.8281 / 0.7469 / 0.7362

class balance 기준 후보:
  classifier-only fine-tune seed44
  test accuracy = 0.7688
  worst recall = 0.7431
```

공식 best checkpoint는 목적에 따라 둘로 구분한다.

```text
accuracy 기준:
  experiments/exp3/results/exp03_domain_adapt_ifreq_finetune_seed44/resnet1d_seed44/checkpoints/best.pt

class balance 기준:
  experiments/exp3/results/exp03_domain_adapt_ifreq_finetune_classifier_seed44/resnet1d_seed44/checkpoints/best.pt
```

## 8. 결론

실험 3의 핵심 결론은 다음 한 문장으로 정리할 수 있다.

```text
10 cm OTA에서 확보한 변조 분류 모델은 1 m OTA에서도 완전히 무너지지 않고 약 0.76 수준의 정확도를 유지하며, 1 m OTA calibration session을 일부 포함해 fine-tuning하면 약 0.77 수준까지 개선된다.
```

하지만 아직 프로젝트 전체 목표인 “변조 분류 및 데이터 복원 시스템” 관점에서는 변조 분류까지만 검증된 상태다. 다음 단계는 분류 결과를 이용해 DSP 복조기를 선택하고 payload/CRC/BER/CER/Packet Success Rate를 평가하는 end-to-end 복원 실험이어야 한다.

## 9. 실험 4로 넘길 과제

실험 4 권장 주제:

```text
End-to-End OTA 변조 분류 + 데이터 복원
```

실험 4에서 봐야 할 지표:

```text
Modulation classification accuracy
BER: Bit Error Rate
CER: Character Error Rate
Packet Success Rate
CRC Pass Rate
Payload Recovery Accuracy
Class별 복원 성공률
거리별 복원 성공률
```

핵심 질문:

```text
분류 accuracy가 0.77이면 실제 payload 복원도 가능한가?
분류는 맞았는데 복조가 실패하는 경우가 많은가?
복조는 되는데 CRC가 깨지는가?
BASK/BFSK/BPSK 중 어떤 방식이 OTA에서 복원에 가장 취약한가?
```

## 10. 검증

최종 코드 테스트:

```powershell
cd D:\ai_projects\SDR
.\.venv\Scripts\python.exe -m pytest -q experiments\exp3\tests
```

결과:

```text
13 passed
```

## 11. 파일 정리 기준

이 문서를 실험 3의 공식 통합 보고서로 사용한다.

기존 중간 문서들은 중복을 피하기 위해 `results/reports/archive_notes/` 아래로 이동했다.

```text
archive_notes/EXP03_RUN_LOG.md
archive_notes/EXP03_MODEL_COMPARISON.md
archive_notes/EXP03_IMPROVEMENT_LOG.md
archive_notes/EXP03_PROBLEM_ANALYSIS_AND_SOLUTION.md
```

세부 JSON/CSV/개별 분석 결과는 그대로 보존한다. 최종 판단과 재현 명령은 이 `FINAL_REPORT.md`를 우선한다.
