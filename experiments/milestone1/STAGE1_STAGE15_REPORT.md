# Oshea2018 Stage1 / Stage1.5 Fixed OTA 실험 보고서

## 1. 실험 개요

`oshea2018` 실험의 목적은 O'Shea 2018 논문식 raw IQ 기반 모델과, 앞선 Exp5/8/9 계열 확장 방법론을 동일한 fixed OTA 데이터셋에서 비교하는 것이다.

비교 대상 모델은 다음 6개로 고정했다.

| 모델 | 입력/방법론 | 목적 |
| --- | --- | --- |
| raw ResNet | raw I/Q `[2, 1024]` | O'Shea 2018 계열 raw IQ ResNet baseline |
| raw VGG | raw I/Q `[2, 1024]` | O'Shea 2018 계열 VGG-style 1D CNN baseline |
| 5ch ResNet | I, Q, magnitude, instantaneous frequency, differential phase | Exp5 계열 feature 확장 |
| MultiTask 5ch | 5ch 입력 + auxiliary heads | Exp8 계열 multitask 학습 |
| MultiTask margin 5ch | 5ch multitask + margin objective | 클래스 간 경계 분리를 강화한 multitask 구조 |
| RF-preprocessed ResNet | RF canonicalization/preprocessed 5ch | Exp9 계열 RF 전처리 비교 |

기존 `raw_ota_clean`, `ota_processed_clean`, `balanced_quick_*` 데이터는 BPSK recall이 0에 가깝고 BPSK가 대부분 BASK로 흡수되는 문제가 있었다. 따라서 기존 clean 데이터는 실패 원인 분석용으로만 보존하고, 공식 비교는 새 fixed OTA 데이터셋만 사용한다.

Stage 역할은 다음과 같다.

| 단계 | 목적 | 데이터 규모 |
| --- | --- | --- |
| Stage1 quick | BPSK collapse, 편향 학습, 데이터 품질 실패를 빠르게 검출 | 17,100 windows |
| Stage1.5 q25 | Stage2보다 작은 1/4 규모로 6개 모델을 모두 학습해 중간 성능 검증 | 135,000 windows |
| Stage2 full | full balanced fixed dataset으로 공식 성능 비교 | 이 보고서 범위 밖 |

## 2. 기존 Clean OTA 실패 분석

기존 clean OTA 데이터에서는 balanced subset까지 만든 뒤에도 BPSK recall이 거의 0에 가까웠다. 대표 confusion matrix는 다음 형태였다.

```text
BASK -> BASK
BFSK -> BFSK
BPSK -> BASK
```

즉 모델이 사실상 BASK/BFSK 2-class 문제처럼 학습했고, BPSK를 독립 클래스로 보지 못했다. 이 현상은 단순 class count 불균형만으로는 설명되지 않았다. balanced dataset으로 session/class count를 맞춰도 BPSK collapse가 유지됐기 때문이다.

주요 판단은 다음과 같다.

| 항목 | 판단 |
| --- | --- |
| 데이터 개수 불균형 | balanced dataset으로 일부 해결했으나 BPSK collapse는 계속 발생 |
| 모델 구조 문제 | raw ResNet, raw VGG, 5ch, multitask 모두에서 유사한 collapse가 나타나 모델 단독 문제로 보기 어려움 |
| RF/수집 품질 문제 | BASK/BPSK pass capture가 적고, BPSK phase transition이 입력 feature에서 충분히 드러나지 않았을 가능성이 큼 |
| 결론 | Stage2 full training 금지, fixed OTA 재수집 필요 |

## 3. SDR 송수신 체인

Fixed OTA 수집의 우선순위는 모델 개선이 아니라 안정적인 SDR 송수신 체인 확보였다. Stage0에서 HackRF TX 단독 송신, RTL-SDR RX 단독 noise capture, RX/TX 동시 capture가 모두 정상 종료되어야 preflight로 넘어가도록 했다.

| 역할 | 장비/방식 | 설명 |
| --- | --- | --- |
| TX | HackRF One + `hackrf_transfer` | 생성한 complex IQ를 CS8 파일로 변환한 뒤 `hackrf_transfer`로 정해진 sample 수만큼 송신 |
| RX | RTL-SDR Blog V4 + SoapySDR | `src.sdr.capture_iq`로 complex64 IQ 저장 |
| 제어 환경 | `radioconda` | SoapySDR, HackRF, RTL-SDR 제어 |
| 학습/분석 환경 | `.venv` | PyTorch 학습, dataset 처리, evaluation |

주요 fixed RF 설정은 다음과 같다.

| 파라미터 | 값 | 의미 |
| --- | ---: | --- |
| `center_freq` | 433,920,000 Hz | RF 중심 주파수 |
| `sample_rate` | 2,400,000 S/s | SDR 송수신 sample rate |
| `target_sample_rate` | 160,000 S/s | channelize/downsample 후 모델 입력용 유효 sample rate |
| `symbol_rate` | 5,000 symbols/s | 변조 symbol rate |
| `samples_per_symbol` | 480 | TX sample rate 기준 symbol당 sample 수 |
| `tx_amp_gain` | 0 | HackRF RF amp off |
| `agc` | false | RX AGC off, 수동 gain 사용 |
| `active_start_seconds` | 1.1 | capture 중 학습/품질 평가에 사용할 active 영역 시작 |
| `active_duration_seconds` | 3.8 | active 영역 길이 |

학습 중 출력된 `FutureWarning`은 PyTorch AMP API 변경 예고다. `torch.cuda.amp.GradScaler`와 `torch.cuda.amp.autocast`가 deprecated 되었고 향후 `torch.amp.*` 형식을 쓰라는 의미다. 현재 학습 실패 원인은 아니며 결과에는 영향을 주지 않는다.

PowerShell에서 `python.exe : ... FutureWarning ... NativeCommandError`처럼 보이는 메시지는 native process stderr 출력이 PowerShell에 에러처럼 표시된 것이다. 아래에 epoch progress가 계속 진행되면 실제 실패가 아니다.

## 4. 데이터 생성 및 수집

모든 송신 capture는 random payload bits를 사용했다. 목적은 모델이 payload pattern 자체를 class shortcut으로 학습하지 못하게 하고, 변조 방식 차이만 학습하도록 하는 것이다.

Payload seed 정책은 다음과 같다.

| 단위 | 정책 | 이유 |
| --- | --- | --- |
| 같은 `session_id + capture_idx` | BASK/BFSK/BPSK가 같은 `payload_seed` 공유 | 세 modulation 간 payload 차이가 class 단서가 되지 않게 함 |
| 다른 session 또는 capture | seed 변경 | 반복 payload memorization 방지 |
| metadata | `random_payload_bits: true`, `payload_seed`, RF 조건 기록 | 재현성과 검증성 확보 |

Preflight는 `tx_vga=25/30/35`, `rx_gain=25/30/35`, `offset=250k/500k`의 총 18개 조건에서 수행했다. 각 조건에서 BASK/BFSK/BPSK를 소량 수집하고 pass rate, SNR, clipping, differential phase 지표를 평가했다.

Preflight 결과 요약:

| 항목 | 결과 |
| --- | --- |
| 총 capture | 108 |
| best condition | `tx30_rx25_off500000` |
| best balanced pass rate | 1.0 |
| best BASK/BPSK pass rate | 1.0 |
| full collection 후보 | 16개 조건 |
| 제외/보류 성격 | BASK/BPSK가 불안정한 조건 |

본수집은 preflight 통과 조건을 사용해 20 sessions로 수행했다.

| 항목 | 값 |
| --- | ---: |
| sessions | 20 |
| session당 noise capture | 1 |
| session당 class별 capture | 10 |
| modulation classes | BASK, BFSK, BPSK |
| 총 class capture | class당 200 |

수집 품질 요약:

| Class | Quality pass rate |
| --- | ---: |
| BASK | 97.5% |
| BFSK | 92.0% |
| BPSK | 100.0% |

Metadata 누락과 payload seed 정책 오류는 발견되지 않았다.

## 5. 데이터 처리

Fixed raw capture는 channelize, downsample, windowing을 거쳐 학습용 `.npz` windows로 변환했다. 이후 session/class balance를 맞춘 dataset과 quick/stage1.5 subset을 생성했다.

| 데이터셋 | 경로 | 용도 |
| --- | --- | --- |
| raw fixed | `data/raw_ota_fixed` | SDR에서 직접 수집한 raw IQ capture |
| processed fixed | `data/ota_processed_fixed` | windowing 완료, 불균형 포함 |
| balanced fixed | `data/ota_processed_fixed_balanced` | Stage2 full 기준 balanced dataset |
| quick subset | `data/ota_processed_fixed_balanced_quick` | Stage1 quick |
| Stage1.5 q25 raw/5ch | `data/ota_processed_fixed_balanced_stage15_q25` | Stage1.5 raw/5ch 학습 |
| Stage1.5 q25 RF | `data/ota_rf_preprocessed_fixed_balanced_stage15_q25` | Stage1.5 RF-preprocessed 학습 |

데이터 규모:

| 데이터셋 | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| processed fixed | 974,892 | 120,972 | 213,480 | 1,309,344 |
| balanced fixed | 378,000 | 72,000 | 90,000 | 540,000 |
| Stage1 quick | 12,600 | 1,800 | 2,700 | 17,100 |
| Stage1.5 q25 | 94,500 | 18,000 | 22,500 | 135,000 |

Stage1.5 q25는 balanced fixed dataset의 1/4 규모다. Split별 session/class balance는 유지했으며, train/val/test session leakage가 없도록 분리했다.

## 6. AI 모델 구조 및 방법론

이 실험의 6개 모델은 크게 두 그룹으로 나뉜다. 첫 번째는 O'Shea 2018 논문 계열의 raw IQ baseline이고, 두 번째는 Exp5/8/9에서 사용한 feature 확장 및 RF 전처리 계열이다. 모든 모델의 최종 출력 class는 `BASK`, `BFSK`, `BPSK` 3개다.

| 모델 | 실제 구현 type | 입력 shape | 핵심 구조 | Loss |
| --- | --- | --- | --- | --- |
| raw ResNet | `oshea2018_resnet1d` | `[2, 1024]` | O'Shea-style residual stack 6개 | cross entropy |
| raw VGG | `oshea2018_vgg1d` | `[2, 1024]` | 7개 Conv-BN-ReLU-MaxPool stack | cross entropy |
| 5ch ResNet | `resnet1d_5ch` | `[5, 1024]` | generic 1D ResNet block stack | cross entropy |
| MultiTask 5ch | `multitask_resnet1d_5ch` | `[5, 1024]` | 5ch ResNet encoder + 3개 head | multitask loss |
| MultiTask margin 5ch | `multitask_resnet1d_margin_5ch` | `[5, 1024]` | MultiTask 5ch + supervised contrastive margin | multitask margin loss |
| RF-preprocessed ResNet | `rf_preprocessed_resnet1d` | `[5, 1024]` | RF 전처리된 5ch 입력 + generic 1D ResNet | cross entropy |

### 공통 입력 단위

모델 입력은 processed dataset의 window 하나다. `window_len=1024`이며, raw IQ 모델은 I/Q 2개 channel만 사용한다. 5ch 계열 모델은 같은 IQ window에서 magnitude, instantaneous frequency, differential phase를 추가 계산하거나 RF-preprocessed dataset에 저장된 5ch 값을 사용한다.

```text
raw IQ input:
  channel 0 = I
  channel 1 = Q

5ch input:
  channel 0 = I
  channel 1 = Q
  channel 2 = magnitude
  channel 3 = instantaneous frequency
  channel 4 = differential phase
```

`magnitude`는 수신 신호의 envelope 변화를 드러낸다. BASK는 amplitude on/off 성격이 강하므로 이 channel이 유용하다. `instantaneous frequency`는 phase 변화율을 나타내며 BFSK의 주파수 이동을 드러낸다. `differential phase`는 인접 sample 간 phase 변화량이므로 BPSK의 0/pi phase flip을 raw I/Q보다 직접적으로 노출한다.

### raw ResNet

raw ResNet은 `OShea2018ResNet1DClassifier`로 구현되어 있다. 입력은 raw I/Q `[2, 1024]`이며, 모델이 별도 RF feature 없이 I/Q waveform에서 직접 변조 특징을 학습한다.

구조는 다음과 같다.

```text
Input [2, 1024]
-> Conv1d(2 -> 32, kernel=7, padding=3)
-> BatchNorm1d(32)
-> ReLU
-> OShea2018ResidualStack x 6
-> Flatten
-> Linear(32 * 16 -> 128)
-> SELU
-> AlphaDropout(0.1)
-> Linear(128 -> 128)
-> SELU
-> AlphaDropout(0.1)
-> Linear(128 -> 3)
```

각 `OShea2018ResidualStack`은 먼저 `MaxPool1d(2)`로 시간축 길이를 절반으로 줄인 뒤, 같은 channel 수 32에서 `Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm`을 수행하고 residual add를 적용한다. 1024 길이 입력은 6번 downsample되어 16이 되므로 classifier의 첫 Linear 입력은 `32 * 16`이다.

이 모델의 장점은 논문 baseline에 가깝고 전처리 가정이 적다는 점이다. 반대로 BPSK phase transition이나 BFSK frequency shift가 raw I/Q 안에서 약하게 드러나면 feature를 전부 CNN이 스스로 찾아야 하므로, 데이터 품질이 낮을 때 collapse에 취약하다. Fixed Stage1.5에서는 데이터 품질이 개선되어 BPSK recall 0.9889까지 회복됐다.

### raw VGG

raw VGG는 `OShea2018VGG1DClassifier`로 구현되어 있다. 입력은 raw I/Q `[2, 1024]`이고, residual connection 없이 convolution block을 깊게 쌓는 VGG-style 구조다.

구조는 다음과 같다.

```text
Input [2, 1024]
-> [Conv1d(ch -> 64, kernel=3, padding=1)
    BatchNorm1d(64)
    ReLU
    MaxPool1d(2)] x 7
-> Flatten
-> Linear(64 * 8 -> 128)
-> SELU
-> AlphaDropout(0.1)
-> Linear(128 -> 128)
-> SELU
-> AlphaDropout(0.1)
-> Linear(128 -> 3)
```

7번의 MaxPool로 1024 길이는 8까지 줄어든다. VGG는 residual path가 없어서 구조적으로 단순하고 해석이 쉽다. 대신 깊이가 증가할수록 정보 손실이 누적될 수 있다. Stage1 quick에서는 subset이 작아 BFSK recall이 0으로 불안정했지만, Stage1.5에서는 BASK/BFSK/BPSK recall이 모두 0.9169 이상으로 회복됐다.

### 5ch ResNet

5ch ResNet은 `ResNet1DClassifier(input_channels=5)`를 사용한다. raw O'Shea ResNet이 32 channel 고정 residual stack을 쓰는 것과 달리, 이 모델은 channel 폭을 32 -> 64 -> 128 -> 256으로 늘리는 generic ResNet 구조다.

구조는 다음과 같다.

```text
Input [5, 1024]
-> Conv block(5 -> 32, kernel=7)
-> ResidualBlock1D(32 -> 32, stride=1)
-> ResidualBlock1D(32 -> 64, stride=2)
-> ResidualBlock1D(64 -> 64, stride=1)
-> ResidualBlock1D(64 -> 128, stride=2)
-> ResidualBlock1D(128 -> 128, stride=1)
-> ResidualBlock1D(128 -> 256, stride=2)
-> AdaptiveAvgPool1d(1)
-> Flatten
-> Dropout
-> Linear(256 -> 3)
```

`ResidualBlock1D`는 `Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm` main path와 skip path를 더한다. channel 수나 stride가 바뀌는 block에서는 skip path에 `1x1 Conv1d + BatchNorm`을 사용해 shape를 맞춘다.

5ch ResNet의 핵심은 모델이 raw I/Q에서 amplitude, frequency, phase 정보를 직접 추론하지 않아도 되게 만드는 것이다. BASK는 magnitude, BFSK는 instantaneous frequency, BPSK는 differential phase에서 각각 더 명확한 단서를 얻는다. Stage1.5에서 worst recall 0.9305로 raw ResNet보다 class 균형성이 좋았다.

### MultiTask 5ch

MultiTask 5ch는 `MultiTaskResNet1DClassifier(input_channels=5)`를 사용한다. Encoder는 5ch ResNet과 같은 residual feature extractor를 사용하지만, classifier 앞에 128차원 embedding을 만들고 3개의 head를 동시에 학습한다.

구조는 다음과 같다.

```text
Input [5, 1024]
-> Conv block + ResidualBlock stack
-> AdaptiveAvgPool1d(1)
-> Flatten
-> Dropout
-> Linear(256 -> 128)
-> ReLU
-> shared embedding z [128]

Heads:
  1. classifier: Linear(128 -> 3)
     target = BASK / BFSK / BPSK

  2. bask_binary_head: Linear(128 -> 2)
     target = BASK vs non-BASK

  3. bfsk_bpsk_head: Linear(128 -> 2)
     target = BFSK vs BPSK, only for non-BASK samples
```

Loss는 다음 세 항의 가중합이다.

```text
loss =
  multiclass_weight * CE(BASK/BFSK/BPSK)
  + bask_binary_weight * CE(BASK vs non-BASK)
  + bfsk_bpsk_weight * CE(BFSK vs BPSK)
```

기본 가중치는 multiclass 1.0, BASK binary 0.45, BFSK/BPSK binary 0.55다. 이 구조는 전체 3-class 분류를 학습하면서 동시에 BASK와 non-BASK의 큰 구분, 그리고 BFSK와 BPSK의 세부 구분을 별도로 압박한다. 이전 clean 데이터에서 BPSK가 BASK로 흡수되던 문제를 감안하면, BASK/non-BASK auxiliary head는 absorption을 낮추는 데 중요한 역할을 한다.

Stage1.5에서는 BPSK recall 0.9872, absorption 0.0149로 매우 안정적이었다.

### MultiTask margin 5ch

MultiTask margin 5ch는 MultiTask 5ch와 같은 network를 사용하지만, loss에 supervised contrastive term을 추가한다. 구현상 `multitask_margin`이거나 모델명이 `_margin_5ch`로 끝나면 contrastive weight를 최소 0.08로 설정한다.

추가되는 supervised contrastive loss의 목적은 embedding 공간에서 같은 class sample은 가깝게, 다른 class sample은 멀게 배치하는 것이다.

```text
embedding z = L2 normalize(z)
similarity = z @ z.T / temperature
positive pair = 같은 class이면서 자기 자신이 아닌 sample
loss = positive pair log-probability를 크게 만들도록 최적화
```

이 방식은 단순히 마지막 classifier만 맞히는 것이 아니라, classifier 앞의 representation 자체를 class별로 더 분리한다. 따라서 BASK/BFSK/BPSK decision boundary가 더 넓어지고, RF 조건이나 payload seed가 바뀌어도 class cluster가 무너지지 않도록 유도한다.

Stage1.5에서 MultiTask margin 5ch는 Accuracy 0.9466, Macro F1 0.9467, Worst recall 0.9341로 전체 1위였다. 특히 worst recall이 가장 높다는 것은 가장 어려운 class에서도 recall이 가장 안정적이라는 뜻이다.

### RF-preprocessed ResNet

RF-preprocessed ResNet은 모델 구조 자체는 `ResNet1DClassifier(input_channels=5)`로 5ch ResNet과 같다. 차이는 입력 dataset이 `data/ota_rf_preprocessed_fixed_balanced_stage15_q25`에서 온다는 점이다. 즉 feature extractor는 같지만, 모델에 들어가기 전 IQ가 RF preprocessing 과정을 거친다.

구조는 다음과 같다.

```text
RF-preprocessed 5ch input [5, 1024]
-> Conv block(5 -> 32, kernel=7)
-> ResidualBlock1D stack
-> AdaptiveAvgPool1d(1)
-> Linear(256 -> 3)
```

이 모델의 목적은 channel/phase/frequency 변동을 전처리 단계에서 줄이고, ResNet이 더 canonical한 representation을 학습하도록 하는 것이다. 다만 전처리가 항상 이득인 것은 아니다. 실제 OTA 신호에서는 전처리가 일부 class-discriminative cue를 약화시킬 수 있고, Stage1.5에서는 6개 모델 중 가장 낮은 accuracy를 보였다.

그럼에도 RF-preprocessed ResNet은 BPSK recall 0.9517, worst recall 0.9212를 기록했다. 따라서 실패 모델은 아니며, clean 데이터에서 보였던 BPSK collapse는 발생하지 않았다. 이 결과는 fixed OTA 데이터 자체가 학습 가능한 형태로 수집되었음을 다시 확인한다.

### 모델별 방법론 차이 요약

| 비교 축 | raw ResNet/VGG | 5ch ResNet | MultiTask 5ch | MultiTask margin 5ch | RF-preprocessed ResNet |
| --- | --- | --- | --- | --- | --- |
| 입력 정보 | I/Q만 사용 | I/Q + RF-derived feature | 5ch | 5ch | 전처리된 5ch |
| phase 정보 | 모델이 직접 추론 | differential phase 제공 | differential phase 제공 | differential phase 제공 + embedding 분리 | 전처리 후 phase 관련 feature 제공 |
| frequency 정보 | 모델이 직접 추론 | instantaneous frequency 제공 | instantaneous frequency 제공 | instantaneous frequency 제공 + embedding 분리 | 전처리 후 frequency 관련 feature 제공 |
| 학습 objective | 3-class CE | 3-class CE | 3-class + 2 auxiliary CE | multitask CE + contrastive margin | 3-class CE |
| 장점 | 논문 baseline, 전처리 가정 적음 | class별 RF 단서 명시 | BASK/non-BASK와 BFSK/BPSK 분해 학습 | class embedding 분리 강화 | RF 변동 보정 시도 |
| 위험 | 데이터 품질 낮으면 collapse 가능 | feature 계산 품질에 의존 | loss 설계 복잡도 증가 | contrastive batch 구성에 영향 | 전처리가 유효 단서를 약화시킬 수 있음 |

## 7. Stage1 Quick 결과

Stage1 quick은 최종 성능 확정용이 아니라 빠른 실패 감지용이다. Gate 기준은 다음과 같다.

```text
BPSK recall >= 0.30
worst recall > 0
BPSK가 confusion matrix에서 전부 BASK로 가지 않을 것
```

`results/fixed_stage1_summary.json` 기준 결과:

| 모델 | Accuracy | Macro F1 | BASK recall | BFSK recall | BPSK recall | Worst recall | Absorption | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MultiTask 5ch | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | pass |
| MultiTask margin 5ch | 0.9996 | 0.9996 | 1.0000 | 1.0000 | 0.9989 | 0.9989 | 0.0006 | pass |
| 5ch ResNet | 0.9996 | 0.9996 | 1.0000 | 1.0000 | 0.9989 | 0.9989 | 0.0006 | pass |
| RF-preprocessed ResNet | 0.6104 | 0.5337 | 0.9889 | 0.0489 | 0.7933 | 0.0489 | 0.5789 | pass by BPSK gate, BFSK risky |
| raw VGG | 0.6667 | 0.5556 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | fail |
| raw ResNet | 0.5348 | 0.4330 | 0.6044 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | fail |

Stage1 quick 결론:

- Fixed data에서는 기존 clean 데이터의 핵심 문제였던 BPSK collapse가 해소됐다.
- 5ch/멀티태스크 계열은 quick subset에서도 매우 강했다.
- raw ResNet/VGG는 quick subset에서는 BFSK recall이 0으로 불안정했지만, Stage1.5에서 데이터가 늘어나자 정상적으로 회복됐다.
- RF-preprocessed ResNet은 BPSK gate는 통과했지만 BFSK recall이 낮아 Stage1 quick에서는 위험 신호가 있었다.

## 8. Stage1.5 Q25 결과

Stage1.5 q25는 Stage2 full balanced dataset의 1/4인 135,000 windows를 사용했다. 목적은 Stage2보다 빠르게 모든 모델을 충분한 데이터로 다시 비교하는 것이다.

Stage1.5 전체 결과:

| 순위 | 모델 | Accuracy | Macro F1 | BASK recall | BFSK recall | BPSK recall | Worst recall | Absorption |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | MultiTask margin 5ch | 0.9466 | 0.9467 | 0.9520 | 0.9341 | 0.9537 | 0.9341 | 0.0324 |
| 2 | MultiTask 5ch | 0.9454 | 0.9459 | 0.9277 | 0.9212 | 0.9872 | 0.9212 | 0.0149 |
| 3 | raw VGG | 0.9445 | 0.9451 | 0.9169 | 0.9260 | 0.9905 | 0.9169 | 0.0045 |
| 4 | 5ch ResNet | 0.9441 | 0.9442 | 0.9391 | 0.9305 | 0.9628 | 0.9305 | 0.0283 |
| 5 | raw ResNet | 0.9436 | 0.9443 | 0.9208 | 0.9212 | 0.9889 | 0.9208 | 0.0096 |
| 6 | RF-preprocessed ResNet | 0.9363 | 0.9367 | 0.9359 | 0.9212 | 0.9517 | 0.9212 | 0.0474 |

Stage1.5 해석:

- BPSK collapse는 해결됐다. 모든 모델의 BPSK recall이 0.9517 이상이다.
- 최고 성능은 MultiTask margin 5ch다. Accuracy, Macro F1, Worst recall 모두 가장 높다.
- raw ResNet/VGG도 fixed data에서는 강하게 회복했다. 이는 이전 clean 데이터 문제가 모델 구조보다 데이터 수집 품질/RF 조건에 가까웠다는 판단을 지지한다.
- 5ch ResNet은 raw 모델과 비슷한 accuracy지만 worst recall이 높아 class 균형성이 좋다.
- RF-preprocessed ResNet은 가장 낮은 accuracy를 보였지만, 모든 class recall이 0.92 이상이므로 실패 모델은 아니다.

## 9. 파라미터 및 지표 의미

### RF 파라미터

| 파라미터 | 의미 |
| --- | --- |
| `sample_rate` | SDR가 송수신하는 초당 complex sample 수 |
| `center_freq` | RF 중심 주파수 |
| `baseband_offset_hz` | DC impairment를 피하기 위해 실제 신호를 중심 주파수에서 떨어뜨리는 offset |
| `tx_vga_gain` | HackRF 송신 VGA gain |
| `tx_amp_gain` | HackRF RF amplifier enable/gain 성격의 설정. 이번 fixed에서는 0 |
| `rx_gain` | RTL-SDR 수신 gain |
| `symbol_rate` | 초당 symbol 수. 변조 transition 밀도에 직접 영향 |
| `active_start_seconds` | capture에서 TX가 안정적으로 들어온 뒤 학습/품질 평가를 시작하는 시간 |
| `active_duration_seconds` | active region 길이 |

### 데이터 파라미터

| 파라미터 | 의미 |
| --- | --- |
| `session_id` | 동일 RF 조건/환경 묶음 |
| `capture_idx` | session 안의 capture 번호 |
| `payload_seed` | random payload bit 생성 seed |
| `random_payload_bits` | payload를 고정 pattern이 아니라 random bits로 생성했는지 여부 |
| `train/val/test split` | 학습/검증/평가 분할 |
| `session leakage` | 같은 session이 train과 test에 동시에 들어가 과대평가를 만드는 문제 |

### 학습 파라미터

| 파라미터 | 값 | 의미 |
| --- | ---: | --- |
| `batch_size` | 512 | 한 step에서 학습하는 windows 수 |
| `epochs` | 8 | 전체 train set 반복 횟수 |
| `learning_rate` | 0.001 | optimizer step size |
| `optimizer` | adam | 학습 최적화 방법 |
| `amp` | true | mixed precision 학습 사용 |
| `num_workers` | 4 | DataLoader worker 수 |
| `pin_memory` | true | GPU 전송을 위한 pinned memory 사용 |
| `persistent_workers` | true | epoch 사이 DataLoader worker 유지 |
| `prefetch_factor` | 4 | worker별 batch prefetch 수 |

### 평가 지표

| 지표 | 의미 |
| --- | --- |
| `accuracy` | 전체 sample 중 맞힌 비율 |
| `macro F1` | class별 F1을 동일 가중치로 평균한 값 |
| class recall | 해당 class sample 중 해당 class로 맞힌 비율 |
| `worst recall` | BASK/BFSK/BPSK recall 중 가장 낮은 값. 특정 class 붕괴를 감지하는 핵심 지표 |
| `absorption` | BFSK/BPSK가 BASK로 흡수되는 비율. BASK 쏠림과 BPSK collapse 감지에 사용 |
| confusion matrix | 실제 class와 예측 class의 대응표 |

## 10. 결론 및 다음 단계

Fixed OTA dataset은 Stage1과 Stage1.5 기준 AI 학습에 적합하다. 기존 clean 데이터에서 발생했던 BPSK collapse는 fixed 재수집 이후 사라졌고, Stage1.5에서는 모든 모델이 BPSK recall 0.95 이상을 달성했다.

Stage1.5 기준 최상위 모델은 MultiTask margin 5ch다. MultiTask 5ch, 5ch ResNet, raw VGG, raw ResNet도 성능 차이가 크지 않아 Stage2 full 비교 대상으로 유지할 가치가 있다. RF-preprocessed ResNet은 가장 낮은 성능이지만 collapse 없이 안정적인 recall을 보였으므로 보조 비교 모델로 유지할 수 있다.

속도 병목은 모델 계산보다 데이터 로딩에 있었다. `.npz` 다량 랜덤 로딩과 HDD I/O가 GPU utilization을 낮췄다. 이후 반복 실험에서는 NVMe 위치에서 dataset/results를 운용하거나, `.npz` 개별 파일 대신 shard/packed dataset 형식을 사용하는 것이 좋다.

Stage2 full 결과는 이 보고서에 포함하지 않는다. Stage2가 완료되면 별도 full comparison report에서 Stage1.5 결과와 분리해 비교해야 한다.
