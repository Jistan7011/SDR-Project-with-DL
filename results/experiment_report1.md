# SDR + PyTorch 변조 분류 실험 보고서

## 1. 실험 개요

본 실험의 목표는 HackRF One으로 BASK, BFSK, BPSK 이진 변조 신호를 송신하고, RTL-SDR Blog V4로 수신한 IQ 데이터를 PyTorch 1D CNN 모델에 입력하여 변조 방식을 분류하는 것이다.

초기 구현 방향은 다음과 같이 잡았다.

- 딥러닝 모델은 변조 방식만 분류한다.
- payload 복원은 분류 결과에 따라 DSP 복조기를 선택하는 구조로 둔다.
- 실제 SDR 실험 전에는 시뮬레이션 IQ 데이터로 학습/평가 파이프라인을 먼저 검증한다.
- Windows SDR 환경은 Radioconda + SoapySDR + Zadig 기반으로 구성한다.

사용 장비와 환경은 다음과 같다.

| 항목 | 값 |
|---|---|
| OS | Windows |
| GPU | NVIDIA GeForce RTX 4060 |
| Python | 3.11.9, project-local `.venv` |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime | 11.8 |
| TX | HackRF One |
| RX | RTL-SDR Blog V4 |
| SDR runtime | `C:\Users\qus70\radioconda` |
| SDR API | SoapySDR |
| 중심 주파수 | 433 MHz |
| TX sample rate | 8 MS/s |
| RX sample rate | 2.4 MS/s |
| Symbol rate | 5 ksps |

## 2. 소프트웨어 파이프라인 구축

처음에는 실제 SDR 없이 시뮬레이션 기반 파이프라인을 구현했다.

구현한 주요 기능은 다음과 같다.

- 프레임 생성: preamble, sync word, payload, CRC-8
- BASK/BFSK/BPSK NumPy 기반 IQ 변조기
- AWGN 추가
- `.npz` 데이터셋 저장
- PyTorch Dataset/DataLoader
- 1D CNN 변조 분류 모델
- 학습/평가 CLI
- confusion matrix, classification report, SNR별 평가 파일 저장
- SoapySDR 기반 HackRF 송신/RTL-SDR 수신 코드

기본 검증 결과:

```text
Python 3.11.9
torch 2.7.1+cu118
torch.version.cuda 11.8
torch.cuda.is_available() True
GPU NVIDIA GeForce RTX 4060
pytest 4 passed
```

시뮬레이션 데이터 학습에서는 최종적으로 약 97% 이상의 test accuracy를 얻었다. 이 결과로 모델, 데이터셋, 학습 루프 자체는 정상임을 확인했다.

## 3. SDR 장비 인식 및 캡처

Radioconda 환경에서 SoapySDR 장비 인식을 확인했다.

```text
HackRF One: detected
RTL-SDR Blog V4: detected
open driver=hackrf: ok
open driver=rtlsdr: ok
```

초기에는 수신과 송신을 수동으로 각각 실행했다. 이 방식은 타이밍을 사람이 맞춰야 하므로, 송신 구간이 수신 파일에 충분히 포함되지 않을 가능성이 있었다. 이후 `run_sdr_capture_sequence` 자동화 명령을 추가하여 다음 순서가 한 번에 실행되도록 변경했다.

```text
1. RTL-SDR 수신 시작
2. 1초 대기
3. HackRF 송신 시작
4. 송신 종료
5. 수신 종료 대기
6. 다음 변조 방식으로 반복
```

자동 캡처 대상:

| 변조 | payload | 파일 |
|---|---|---|
| BASK | A | `data\real\raw_iq\bask_a.bin` |
| BFSK | F | `data\real\raw_iq\bfsk_f.bin` |
| BPSK | P | `data\real\raw_iq\bpsk_p.bin` |

최신 캡처 파일 크기는 각각 `192,000,000 bytes`이며, 이는 `complex64` 기준 19,200,000 samples, 8초 수신에 해당한다.

## 4. 주요 시행착오

### 4.1 시뮬레이션 모델을 real IQ에 바로 적용

처음에는 시뮬레이션 데이터로 학습한 `best.pt`를 실제 SDR 캡처에 바로 적용했다.

결과:

```text
accuracy = 0.3333
```

분석 결과, 모델이 모든 real window를 한 클래스(BASK 또는 BFSK)로 예측하는 현상이 나타났다.

원인으로 판단한 내용:

- 시뮬레이션 IQ와 실제 RF IQ의 분포 차이가 큼
- 실제 캡처에는 DC offset, gain 차이, 주파수 offset, 외부 톤, 장비 특성이 포함됨
- 송신 신호가 충분히 강하지 않거나, 수신 window가 송신 구간을 잘 포함하지 않을 가능성

### 4.2 TX gain 0 문제

초기 `config.yaml`에는 `tx_gain: 0`만 있었다. SoapySDR HackRF TX gain 항목을 확인해 보니 다음 gain 이름이 있었다.

```text
HackRF TX gains: LNA, AMP, VGA
RTL-SDR RX gains: TUNER
```

따라서 generic gain 대신 HackRF의 `AMP`, `VGA` gain을 명시적으로 제어하도록 코드를 수정했다.

변경 후:

```yaml
tx_amp_gain: 0
tx_vga_gain: 20
```

그리고 실험 명령에서는 `--tx-vga-gain`을 직접 지정했다.

### 4.3 TX gain별 변화

TX VGA gain을 바꾸며 real capture 평가를 비교했다.

| 조건 | 결과 |
|---|---:|
| 기존 낮은 gain 또는 시뮬레이션 모델 직접 적용 | 0.3333 |
| `--tx-vga-gain 30` | 0.4611 |
| `--tx-vga-gain 40` | 0.4044 |

해석:

- gain 30에서 정확도가 상승했으므로 실제 송신 신호가 더 명확하게 반영되기 시작했다.
- gain 40에서는 오히려 떨어졌으므로 과입력, 왜곡, 수신기 포화 가능성이 있다.
- 이후 실험 조건은 gain 30을 기준으로 잡았다.

### 4.4 캡처 분석에서 강한 외부 피크 발견

raw capture 분석 도구를 추가해 RMS와 spectrum을 확인했다.

이전 분석 결과, 세 파일 모두 약 `+910 kHz` 근처에 강한 피크가 있었다.

| 파일 | peak freq |
|---|---:|
| `bask_a.bin` | 약 +910.28 kHz |
| `bfsk_f.bin` | 약 +910.25 kHz |
| `bpsk_p.bin` | 약 +910.24 kHz |

이 피크는 송신 변조 신호의 목표 특징보다 강한 외부 신호 또는 원치 않는 톤일 수 있다고 판단했다. 따라서 import 단계에 channel filter 및 frequency shift 옵션을 추가했다.

다만 이후 실험에서는 먼저 전체 real 캡처를 학습 가능한 형태로 만드는 것이 더 중요하다고 판단하여, active 송신 구간에서 window를 랜덤 샘플링하는 방식으로 개선했다.

### 4.5 시간순 train/val/test split 문제

초기 real fine-tuning에서는 같은 캡처 파일에서 다음처럼 시간 구간을 나눴다.

```text
train: 1.1s ~ 2.9s
val:   2.9s ~ 3.4s
test:  3.4s ~ 3.9s
```

이 방식으로 학습했을 때 다음 현상이 나타났다.

```text
train_loss -> 거의 0까지 감소
val_accuracy -> 0.3333에 고착
```

즉 모델은 train 구간을 외우지만, 바로 뒤 시간 구간인 val/test에는 일반화하지 못했다. 이는 실제 RF 캡처에서 시간에 따른 gain, 위상, 주파수 drift, 송신 시작/종료 구간 차이가 커서 생긴 문제로 판단했다.

### 4.6 랜덤 window split으로 변경

위 문제를 해결하기 위해 `import_real_sequence`를 추가했다. 이 방식은 각 modulation 파일의 active TX 구간 전체에서 window offset을 랜덤으로 뽑고, 그 window들을 train/val/test로 나눈다.

변경 전:

```text
연속 시간 구간 단위 split
```

변경 후:

```text
active 송신 구간 전체에서 random window sampling
```

생성된 real dataset:

| split | samples |
|---|---:|
| train | 1890 |
| val | 405 |
| test | 405 |

이 변경 후 validation accuracy가 크게 개선되었다.

중간 결과:

```text
best val accuracy = 0.7975
test accuracy = 0.7432
```

이후 더 큰 real random window dataset으로 재학습한 최신 결과는 다음과 같다.

```text
best val accuracy = 0.8207
test accuracy = 0.8193
```

## 5. 최종 결과

최신 평가 결과:

```text
accuracy = 0.8192592593
macro f1 = 0.8189116058
```

클래스별 성능:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BASK | 0.7677 | 0.8667 | 0.8142 | 225 |
| BFSK | 0.8789 | 0.7422 | 0.8048 | 225 |
| BPSK | 0.8268 | 0.8489 | 0.8377 | 225 |

Confusion matrix:

| True \ Pred | BASK | BFSK | BPSK |
|---|---:|---:|---:|
| BASK | 195 | 12 | 18 |
| BFSK | 36 | 167 | 22 |
| BPSK | 23 | 11 | 191 |

해석:

- BASK와 BPSK는 recall이 0.85 수준으로 비교적 잘 잡힌다.
- BFSK는 precision은 높지만 recall이 상대적으로 낮다.
- BFSK 일부가 BASK 또는 BPSK로 섞인다.
- 전체적으로 실제 SDR 캡처 기반 3-class 변조 분류가 가능함을 확인했다.

## 6. 방법 변경에 따른 결과 요약

| 단계 | 방법 | 결과 | 판단 |
|---|---|---:|---|
| 1 | 시뮬레이션 학습/평가 | 약 0.97 | 코드/모델 파이프라인 정상 |
| 2 | 시뮬레이션 모델을 real 캡처에 직접 적용 | 0.3333 | domain gap 큼 |
| 3 | 수동 송수신 캡처 | 불안정 | 타이밍 문제 가능 |
| 4 | 자동 송수신 캡처 추가 | 캡처 안정화 | 재현성 개선 |
| 5 | HackRF TX gain 0 | 0.3333 근처 | 송신 신호 약함 |
| 6 | TX VGA 30 | 0.4611 | 신호 반영 개선 |
| 7 | TX VGA 40 | 0.4044 | 과입력/왜곡 가능 |
| 8 | 시간순 real split fine-tuning | train loss 0, val 0.3333 | 시간 구간 편향 |
| 9 | active 구간 random window split | 0.7432 | real 학습 가능 확인 |
| 10 | 최신 random real dataset 재학습 | 0.8193 | 실험 성공 기준 도달 |

## 7. 현재 한계

현재 결과는 실제 SDR 데이터로 의미 있는 분류 성능을 얻었지만, 다음 한계가 있다.

- 같은 캡처 세션 안에서 random window를 나눴기 때문에 완전한 세션 독립 평가는 아니다.
- BASK/BFSK/BPSK 각각 payload가 하나씩만 사용되었다.
- center frequency, gain, attenuator, 주변 RF 환경이 고정된 조건이다.
- 실제 수신 IQ에서 payload 복원까지 안정적으로 검증한 단계는 아니다.
- 주파수 offset, DC offset, 수신기 gain 변화에 대한 robustness는 아직 제한적이다.

## 8. 다음 실험 계획

다음 단계에서는 세션 단위 일반화 성능을 확인해야 한다.

권장 계획:

1. 동일 조건에서 BASK/BFSK/BPSK를 각 5회 이상 반복 캡처한다.
2. `session_001`, `session_002`처럼 파일을 분리한다.
3. train/test를 window 단위가 아니라 session 단위로 나눈다.
4. payload를 A/F/P 고정에서 random payload로 확장한다.
5. TX gain 25, 30, 35 조건을 비교한다.
6. RX gain 20, 30, 40 조건을 비교한다.
7. BPSK/BFSK 혼동을 줄이기 위해 carrier offset 보정 또는 spectrogram 기반 입력을 추가 검토한다.

## 9. 결론

본 실험에서는 Windows + Radioconda + SoapySDR 환경에서 HackRF One 송신과 RTL-SDR Blog V4 수신을 이용하여 실제 RF IQ 데이터를 수집했고, PyTorch 1D CNN 기반 변조 분류 모델을 학습했다.

초기에는 시뮬레이션 모델이 실제 캡처에 전혀 일반화되지 않아 0.3333 수준에 머물렀다. 이후 SDR 자동 캡처, HackRF gain 명시 제어, real IQ import, capture 분석, random window split을 차례로 적용하면서 최종적으로 실제 SDR test set에서 약 0.8193 accuracy를 얻었다.

따라서 현재 단계의 결론은 다음과 같다.

```text
시뮬레이션 기반 모델만으로는 실제 SDR IQ에 바로 적용하기 어렵다.
하지만 실제 캡처 데이터를 이용한 fine-tuning과 적절한 window sampling을 적용하면
BASK/BFSK/BPSK 3-class 변조 분류가 가능하다.
```
