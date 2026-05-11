# Experiment 5 Final Report

## 1. 실험 목적

실험 5의 목적은 실험 4에서 확인된 주요 병목인 `BFSK/BPSK -> BASK` 오분류를 줄이는 것이다. 실험 4에서는 end-to-end 복원 성능의 병목이 DSP 복조기보다는 변조 분류기 쪽에 있음을 확인했다. 특히 BASK는 비교적 잘 맞지만 BFSK와 BPSK가 BASK로 흡수되는 경향이 있었다.

따라서 실험 5는 복원 파이프라인 개선이 아니라, 분류기 자체의 `BFSK recall`, `BPSK recall`, `worst-class recall` 개선에 집중했다.

## 2. 실험 조건

- RF path: OTA antenna
- RF cable between SDR: false
- TX: HackRF One
- RX: RTL-SDR Blog V4
- 거리: 약 1 m
- center frequency: 433 MHz
- TX sample rate: 8 MS/s
- RX sample rate: 2.4 MS/s
- symbol rate: 5 ksps
- modulation classes: `BASK`, `BFSK`, `BPSK`
- payload pool: `A, F, P, 0, 1, 7, K, R, S, Z`

입력 window는 실험 4의 1024 sample에서 2048 sample로 확장했다. 160 kS/s target sample rate와 5 ksps symbol rate 기준으로 symbol당 약 32 sample이므로, 2048 sample은 약 64 symbol을 포함한다. 이는 40-bit frame 전체를 더 안정적으로 포함시키기 위한 변경이다.

입력 feature는 기존 `[I,Q,instantaneous_frequency]`에서 다음 5채널로 확장했다.

```text
[I, Q, magnitude, instantaneous_frequency, differential_phase]
```

`differential_phase`는 BPSK의 위상 전이 정보를 더 직접적으로 보이게 하기 위해 추가했다.

## 3. 데이터 구성

실험 5는 exp4 raw IQ 15 session을 read-only reference로 사용하고, exp5 신규 15 session을 추가 수집했다.

- exp4 reference sessions: `session_001 ~ session_015`
- exp5 new sessions: `session_016 ~ session_030`
- 총 session 수: 30
- processed samples: 18,000
- train: 12,600
- val: 3,000
- test: 2,400

split은 session-held-out 방식으로 구성했다.

| split | sessions | samples |
| --- | --- | ---: |
| train | `session_001~012`, `session_016~024` | 12,600 |
| val | `session_013~015`, `session_025~026` | 3,000 |
| test | `session_027~030` | 2,400 |

각 split은 class-balanced 상태다.

| split | BASK | BFSK | BPSK |
| --- | ---: | ---: | ---: |
| train | 4,200 | 4,200 | 4,200 |
| val | 1,000 | 1,000 | 1,000 |
| test | 800 | 800 | 800 |

## 4. 학습 및 비교 모델

비교한 모델은 다음 두 계열이다.

- `ResNet1D`
- `FusionResNet1D`

ResNet1D는 실험 3~4에서 가장 안정적인 기준 모델이었기 때문에 baseline 겸 1차 후보로 사용했다. FusionResNet1D는 time-domain branch와 spectral branch를 함께 쓰는 feature fusion 구조로, 논문에서 제안되는 feature fusion 방향을 반영한 비교 후보였다.

두 모델 모두 seed `42`, `43`, `44`로 학습했다. checkpoint 선택 기준은 overall accuracy보다 worst recall을 더 중시하는 balanced score로 설정했다.

## 5. 결과

### 5.1 ResNet1D 결과

| model | seed | accuracy | macro F1 | BASK recall | BFSK recall | BPSK recall | worst recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet1D | 42 | 0.7308 | 0.7317 | 0.7588 | 0.7288 | 0.7050 | 0.7050 |
| ResNet1D | 43 | 0.7192 | 0.7190 | 0.7688 | 0.7125 | 0.6763 | 0.6763 |
| ResNet1D | 44 | 0.7233 | 0.7235 | 0.7375 | 0.7400 | 0.6925 | 0.6925 |
| ResNet1D 평균 | - | 0.7244 | 0.7247 | 0.7550 | 0.7271 | 0.6913 | 0.6913 |

가장 좋은 단일 모델은 `ResNet1D seed42`다.

```text
Confusion matrix, ResNet1D seed42

true\pred   BASK  BFSK  BPSK
BASK         607   109    84
BFSK         156   583    61
BPSK         147    89   564
```

### 5.2 FusionResNet1D 결과

| model | seed | accuracy | macro F1 | BASK recall | BFSK recall | BPSK recall | worst recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FusionResNet1D | 42 | 0.7275 | 0.7288 | 0.6325 | 0.8725 | 0.6775 | 0.6325 |
| FusionResNet1D | 43 | 0.7271 | 0.7287 | 0.9100 | 0.6450 | 0.6262 | 0.6262 |
| FusionResNet1D | 44 | 0.7288 | 0.7294 | 0.7362 | 0.8325 | 0.6175 | 0.6175 |
| FusionResNet1D 평균 | - | 0.7278 | 0.7290 | 0.7596 | 0.7833 | 0.6404 | 0.6254 |

FusionResNet1D는 평균 accuracy는 ResNet1D보다 약간 높지만, seed별 class recall 변동이 매우 크다. 특히 seed42는 BFSK recall이 높지만 BASK recall이 크게 낮고, seed43은 BASK recall은 높지만 BFSK/BPSK recall이 낮다. 따라서 안정적인 공식 best로 쓰기에는 부적합하다.

## 6. Calibration 결과

`ResNet1D seed42`에 대해 validation set 기준 class bias calibration을 수행했다.

| class | logit bias |
| --- | ---: |
| BASK | 0.000 |
| BFSK | 0.000 |
| BPSK | 0.000 |

calibration은 어떤 class bias도 선택하지 않았다. 즉 현재 문제는 단순히 BASK logit bias를 낮추면 해결되는 문제가 아니다. feature와 decision boundary 자체가 아직 BFSK/BPSK와 BASK를 충분히 분리하지 못한 상태로 해석된다.

Calibration 후 test 결과는 원본과 동일했다.

| split | accuracy | BASK recall | BFSK recall | BPSK recall | worst recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.7263 | 0.7450 | 0.7140 | 0.7200 | 0.7140 |
| test | 0.7308 | 0.7588 | 0.7288 | 0.7050 | 0.7050 |

## 7. 실험 4 대비 변화

실험 4 최종 분류기는 다음 성능이었다.

| experiment | accuracy | BASK recall | BFSK recall | BPSK recall |
| --- | ---: | ---: | ---: | ---: |
| exp4 | 0.7178 | 0.8483 | 0.6683 | 0.6367 |
| exp5 best, ResNet1D seed42 | 0.7308 | 0.7588 | 0.7288 | 0.7050 |
| change | +0.0131 | -0.0896 | +0.0604 | +0.0683 |

실험 5는 BFSK/BPSK recall을 개선했다.

- BFSK recall: `0.6683 -> 0.7288`
- BPSK recall: `0.6367 -> 0.7050`
- accuracy: `0.7178 -> 0.7308`

하지만 BASK recall은 감소했다.

- BASK recall: `0.8483 -> 0.7588`

따라서 실험 5는 BFSK/BPSK 개선 방향은 맞았지만, 전체 성공 기준에는 도달하지 못했다.

## 8. 성공 기준 평가

| criterion | target | result | pass |
| --- | ---: | ---: | --- |
| overall accuracy | >= 0.78 | 0.7308 | fail |
| BFSK recall | >= 0.75 | 0.7288 | fail |
| BPSK recall | >= 0.75 | 0.7050 | fail |
| worst recall | >= 0.74 | 0.7050 | fail |
| BASK recall | >= 0.75 | 0.7588 | pass |

실험 5는 최종 성공 기준 기준으로는 실패다. 다만 exp4 대비 BFSK/BPSK recall이 각각 약 6~7%p 개선되었으므로, feature 확장과 window 확장은 일부 효과가 있었다.

## 9. 오류 분석

Best 모델인 ResNet1D seed42의 주요 오분류는 여전히 `BFSK/BPSK -> BASK`다.

```text
BFSK -> BASK: 156 / 800
BPSK -> BASK: 147 / 800
BFSK/BPSK -> BASK total: 303 / 1600
BFSK/BPSK -> BASK rate: 0.1894
```

반면 BFSK와 BPSK 사이의 직접 혼동도 존재한다.

```text
BFSK -> BPSK: 61 / 800
BPSK -> BFSK: 89 / 800
```

가장 큰 문제는 BFSK/BPSK끼리만 구분하지 못하는 것이 아니라, 먼저 non-BASK 신호를 BASK와 분리하는 단계에서 상당한 오류가 난다는 점이다.

## 10. 해석

실험 5에서 2048 window와 5채널 feature는 BFSK/BPSK recall을 개선했다. 그러나 단일 3-class softmax 구조에서는 BASK와 non-BASK의 decision boundary가 여전히 불안정하다.

FusionResNet1D는 seed별로 특정 class에 강하게 치우치는 경향을 보였다. 이는 spectral branch가 어떤 seed에서는 BFSK를 과도하게 선호하고, 다른 seed에서는 BASK를 과도하게 선호한다는 뜻이다. 현재 데이터 규모와 architecture에서는 FusionResNet1D를 공식 best로 채택하기 어렵다.

Calibration이 bias를 0으로 둔 것도 중요하다. 단순 후처리 bias로 해결될 문제가 아니라, 모델 구조 또는 학습 task를 바꿔야 한다.

## 11. 결론

실험 5의 공식 best 모델은 `ResNet1D seed42`다.

```text
Best checkpoint:
experiments/exp5/results/exp05_resnet_2048/resnet1d_seed42/checkpoints/best.pt
```

실험 5는 다음을 달성했다.

- exp4 대비 accuracy 소폭 개선
- exp4 대비 BFSK recall 개선
- exp4 대비 BPSK recall 개선
- 2048 window와 differential phase feature의 유효성 일부 확인

하지만 다음은 달성하지 못했다.

- 최소 accuracy 0.78
- BFSK recall 0.75
- BPSK recall 0.75
- worst recall 0.74
- BFSK/BPSK -> BASK 오류의 충분한 감소

따라서 실험 5는 `부분 개선, 최종 기준 미달`로 판정한다.

## 12. 다음 개선 방향

다음 단계는 단일 3-class classifier를 계속 학습시키는 것보다 2-stage classifier를 검증하는 것이 타당하다.

```text
Stage 1:
  BASK vs non-BASK

Stage 2:
  non-BASK subset에서 BFSK vs BPSK
```

이유는 현재 오류가 BFSK/BPSK 간 혼동만이 아니라, BFSK/BPSK가 BASK로 흡수되는 형태이기 때문이다. 먼저 BASK와 non-BASK를 분리하고, 그 다음 BFSK/BPSK를 분리하면 현재 병목에 더 직접적으로 대응할 수 있다.

후속 실험에서는 다음 항목을 우선한다.

- 2-stage classifier
- BASK vs non-BASK binary objective
- BFSK vs BPSK binary objective
- stage별 confidence와 reject threshold
- exp5 best ResNet1D를 feature extractor로 재사용
- end-to-end 복원 성능 재평가

## 13. 주요 산출물

- Dataset manifest: `data/processed/manifest_exp05_classification.json`
- Best checkpoint: `results/exp05_resnet_2048/resnet1d_seed42/checkpoints/best.pt`
- Best eval metrics: `results/exp05_resnet_2048/resnet1d_seed42/logs/eval_metrics.json`
- Calibration report: `results/exp05_resnet_2048/resnet1d_seed42/calibration/exp05_class_bias_calibration.md`
- Error analysis: `results/analysis/resnet_seed42/exp05_bfsk_bpsk_error_summary.md`
- Confusion matrix figure: `results/analysis/resnet_seed42/exp05_focus_confusion.png`
