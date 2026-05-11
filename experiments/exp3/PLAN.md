# 실험 3 계획: 1 m OTA 거리 일반화 + Feature Fusion

작성일: 2026-05-05

> 참고: 이 파일은 실험 3 시작 시점의 계획서다. 실제 진행 과정, 시행착오, 최종 결과, 채택/보류 판단은 `FINAL_REPORT.md`에 통합 정리했다.

## 1. 목적

실험 3의 목적은 실험 1, 2의 `10 cm OTA 안테나 송수신` 조건에서 얻은 변조 분류 모델이 `1 m OTA 안테나 송수신` 조건에서도 일반화되는지 검증하는 것이다. 두 SDR 사이에는 RF 케이블이 없다. HackRF One과 RTL-SDR V4는 같은 PC에 USB로 연결되어 있고, RF 신호는 안테나를 통해 공중으로 전송된다.

실험 3은 단순히 모델을 바꾸는 실험이 아니라, `10 cm -> 1 m` 거리 변화가 만드는 domain shift를 측정하는 실험이다.

## 2. 실험 1, 2 해석 수정

실험 1과 실험 2는 모두 유선 RF 연결 실험이 아니라 OTA 안테나 실험이다.

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
  result: ResNet1D ensemble accuracy 0.7594
  해석: 10 cm OTA session-held-out generalization
```

실험 3은 이 기준을 `1 m OTA`로 확장한다.

## 3. 논문별 반영 사항

- O'Shea 2016, Convolutional Radio Modulation Recognition Networks:
  raw complex IQ를 `[I,Q]` 2채널 time series로 CNN에 넣는 현재 구조는 타당하다. 다만 실제 RF channel에서는 AWGN 외에 CFO, timing offset, gain scale, multipath, phase drift가 생기므로 feature와 metadata를 늘린다.

- O'Shea 2016, Learning to Communicate:
  autoencoder 통신 자체는 실험 3 범위에서 제외한다. 대신 channel impairment를 학습/평가 조건에 명시해야 한다는 관점을 반영한다. payload recovery, CRC pass, BER은 2차 평가 후보로 둔다.

- O'Shea 2016, Radio Transformer Networks:
  1 m OTA에서는 frequency offset, phase drift, time shift, sample clock mismatch 영향이 커질 수 있다. 실험 3에서는 channelize/downsample/normalize를 유지하고, 실패 시 실험 4 후보를 RTN-style synchronization으로 둔다.

- O'Shea 2017, Semi-Supervised Radio Signal Identification:
  unknown signal과 noise-only 문제는 중요하지만, 실험 3의 1차 목표는 supervised 3-class classification이다. open-set, reject option, semi-supervised embedding은 후속 실험 후보로 둔다.

- O'Shea 2018, Over-the-Air Deep Learning Based Radio Signal Classification:
  실험 3의 핵심 근거다. `[2,1024]` IQ window, OTA measurement, ResNet 비교, condition-wise 평가 방향을 유지한다. DC impairment를 피하기 위해 100 kHz offset은 primary로 쓰지 않고 250 kHz 또는 500 kHz를 사용한다.

- Zhou 2019, Robust Modulation Classification Using CNN:
  overall accuracy만 보지 않고 SNR별, class별, condition별 robustness를 평가한다. confusion matrix, class recall, SNR bin accuracy, session별 accuracy를 필수 산출물로 둔다.

- Han 2021, Deep Feature Fusion:
  실험 3의 핵심 모델 근거다. raw IQ 단독이 아니라 magnitude, instantaneous frequency, PSD/spectral feature를 결합한다.

- Zhang 2021, Attention Mechanism and Hybrid Parallel NN:
  CNN local feature와 temporal feature 결합은 후속 방향으로 타당하다. 실험 3에서는 먼저 feature fusion만 검증하고 CNN-GRU/Attention은 실험 4 후보로 둔다.

- 2025 한국어 RF AMC 리뷰:
  실험별 dataset 관리, metadata 관리, 모델 비교 자동화, SNR/채널 왜곡 반영의 필요성을 반영한다.

## 4. Session 설계

신규 session은 `session_016 ~ session_021`로 수집한다. 모두 `1 m OTA` 조건이다.

```text
session_016:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 30
  rx_gain: 30
  baseband_offset_hz: 500000

session_017:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 25
  rx_gain: 30
  baseband_offset_hz: 500000

session_018:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 35
  rx_gain: 30
  baseband_offset_hz: 500000

session_019:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 30
  rx_gain: 25
  baseband_offset_hz: 500000

session_020:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 30
  rx_gain: 35
  baseband_offset_hz: 500000

session_021:
  distance: 1.0 m
  antenna_layout: face_to_face
  tx_vga_gain: 30
  rx_gain: 30
  baseband_offset_hz: 250000
```

각 session에는 다음 metadata를 저장한다.

```json
{
  "rf_path": "ota_antenna",
  "rf_cable_between_sdr": false,
  "tx_usb_connected_to_pc": true,
  "rx_usb_connected_to_pc": true,
  "tx_rx_distance_m": 1.0,
  "antenna_layout": "face_to_face",
  "tx_antenna_orientation": "fixed",
  "rx_antenna_orientation": "fixed",
  "line_of_sight": true,
  "near_metal_objects": "minimized",
  "human_nearby": "minimized"
}
```

## 5. Dataset Split

```text
train:
  exp2 session_001 ~ session_009

val:
  exp2 session_010 ~ session_012

test-A:
  exp2 session_013 ~ session_015
  목적: exp2 결과 재현 및 10 cm OTA 기준 비교

test-B:
  exp3 session_016 ~ session_021
  목적: 1 m OTA 거리 일반화 검증
```

같은 session의 window가 여러 split에 들어가면 실패로 처리한다.

## 6. 입력 Feature 및 모델

비교 baseline은 exp2 공식 ResNet1D `[I,Q]`다.

```text
baseline input:
  [batch, 2, 1024]
```

실험 3 후보는 Feature Fusion ResNet1D다.

```text
time-domain branch:
  input [I,Q,magnitude,instantaneous_frequency]
  shape [batch, 4, 1024]
  -> ResNet1D-style residual blocks
  -> time embedding

spectral branch:
  FFT power spectrum from I/Q
  -> adaptive pooling to fixed spectral bins
  -> small MLP
  -> spectral embedding

fusion:
  concat(time embedding, spectral embedding)
  -> dropout
  -> linear classifier
  -> logits [BASK, BFSK, BPSK]
```

Feature ablation 순서는 다음으로 둔다.

```text
1. [I,Q]
2. [I,Q,magnitude]
3. [I,Q,instantaneous_frequency]
4. [I,Q,magnitude,instantaneous_frequency]
5. FusionResNet1D
```

## 7. 실행 명령

기본 위치:

```powershell
cd D:\ai_projects\SDR\experiments\exp3\sourcecode
```

session capture:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp03_capture_session --session-id session_016 --distance-m 1.0 --antenna-layout face_to_face --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000
```

dataset import:

```powershell
Remove-Item -Recurse -Force ..\data\processed -ErrorAction SilentlyContinue
..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp03_sessions --include-exp2-reference --min-new-sessions 6
```

seed sweep:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp03.yaml --model-type fusion_resnet1d --output-root ..\results\exp03_fusion_resnet --eval-split test_b
```

analysis:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp03_errors --config ..\config\config.exp03.yaml --data-root ..\data\processed --output-dir ..\results\reports
..\..\..\.venv\Scripts\python.exe -m src.analysis.make_exp03_report_figures --report-dir ..\results\reports --output-dir ..\results\figures\final_report
```

## 8. 성공 기준

최소 성공 기준:

```text
test-A accuracy >= 0.7594 근처
test-B accuracy >= 0.70
BFSK recall >= 0.73
BPSK recall >= 0.68
특정 class recall < 0.60 collapse 없음
```

목표 성공 기준:

```text
test-A accuracy >= 0.78
test-B accuracy >= 0.75
worst class recall >= 0.68
BFSK->BASK + BPSK->BASK 오류가 exp2 baseline 대비 15% 이상 감소
```

실패 기준:

```text
FusionResNet1D가 test-A와 test-B 모두에서 ResNet1D [I,Q]보다 낮음
또는 test-B에서 특정 class recall이 0.60 미만
또는 1 m OTA 신규 session 전체가 한 class로 collapse
```

## 9. Test Plan

```powershell
cd D:\ai_projects\SDR\experiments\exp3\sourcecode
..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

검증 항목:

- exp3 split 함수가 `train/val/test-A/test-B`를 올바르게 나누는지 확인
- `test-B`에 `session_016` 이후 session만 들어가는지 확인
- feature mode별 shape 확인
- `FusionResNet1D` forward pass가 logits `[batch,3]`을 반환하는지 확인
- exp3 capture CLI가 dry-run에서 올바른 RX/TX 명령을 생성하는지 확인
