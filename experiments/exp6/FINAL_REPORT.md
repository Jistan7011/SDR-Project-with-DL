# Experiment 6 Final Report

## 요약

실험 6은 exp5에서 남은 핵심 오류인 `BFSK/BPSK -> BASK` 오분류를 줄이기 위해 단일 3-class softmax를 2-stage classifier로 분해해 검증한 실험이다.

- Stage 1: `BASK` vs `NON_BASK`
- Stage 2: `BFSK` vs `BPSK`
- 데이터: exp5 processed dataset 재사용
- 신규 OTA 수집: 없음
- 입력: exp5와 동일한 `[I,Q,magnitude,instantaneous_frequency,differential_phase]`, window size `2048`
- 기준 비교: exp5 best ResNet1D seed42

## 구현 및 실행 상태

구현과 공식 실행은 완료되었다.

- Unit tests: `27 passed`
- Stage 1 binary dataset: train `12600`, val `3000`, test `2400`
- Stage 2 binary dataset: train `8400`, val `2000`, test `1600`
- Stage 1 full train: 80 epoch 완료
- Stage 2 full train: 80 epoch 완료
- Two-stage threshold sweep/evaluation 완료
- Error analysis 완료

## Exp5 기준선

| metric | exp5 best |
| --- | ---: |
| accuracy | 0.7308 |
| BASK recall | 0.7588 |
| BFSK recall | 0.7288 |
| BPSK recall | 0.7050 |
| worst recall | 0.7050 |
| `BFSK/BPSK -> BASK` rate | 0.1894 |

## Exp6 공식 결과

Stage 1 threshold sweep 결과 공식 threshold는 `0.500`으로 선택되었다.

| split | accuracy | macro F1 | BASK recall | BFSK recall | BPSK recall | worst recall | `BFSK/BPSK -> BASK` rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.7233 | 0.7234 | 0.7920 | 0.6720 | 0.7060 | 0.6720 | 0.1990 |
| test | 0.7238 | 0.7244 | 0.8187 | 0.6512 | 0.7013 | 0.6512 | 0.2338 |

Test confusion matrix:

| actual \ predicted | BASK | BFSK | BPSK |
| --- | ---: | ---: | ---: |
| BASK | 655 | 49 | 96 |
| BFSK | 192 | 521 | 87 |
| BPSK | 182 | 57 | 561 |

## 성공 기준 판정

| criterion | target | result | pass |
| --- | ---: | ---: | --- |
| accuracy | >= 0.76 | 0.7238 | False |
| BFSK recall | >= 0.76 | 0.6512 | False |
| BPSK recall | >= 0.74 | 0.7013 | False |
| worst recall | >= 0.72 | 0.6512 | False |
| BASK recall | >= 0.72 | 0.8187 | True |
| `BFSK/BPSK -> BASK` reduction | >= 25% | -23.42% | False |

실험 6은 최소 성공 기준을 만족하지 못했다.

## 해석

2-stage 구조는 의도와 반대로 BASK recall을 높이는 대신 `BFSK/BPSK -> BASK` 오류를 늘렸다. exp5에서는 해당 오류율이 `0.1894`였지만, exp6 test에서는 `0.2338`로 악화되었다.

Stage 2 자체는 `BFSK`와 `BPSK`를 어느 정도 구분할 수 있었다. 그러나 Stage 1에서 non-BASK 샘플을 BASK로 흡수하는 비율이 커지면서 Stage 2까지 도달하지 못하는 BFSK/BPSK 샘플이 많아졌다. 따라서 현재 병목은 Stage 2의 BFSK/BPSK 분리보다는 Stage 1의 `BASK` vs `NON_BASK` decision boundary에 있다.

Threshold sweep에서도 trade-off가 명확했다.

- threshold를 낮추면 `BFSK/BPSK -> BASK` 오류는 줄지만 BASK recall이 크게 무너진다.
- threshold를 높이면 BASK recall은 좋아지지만 BFSK/BPSK가 BASK로 흡수된다.
- 설정된 BASK recall 하한 `0.72`를 만족하면서 오류 감소까지 동시에 달성하는 threshold가 없었다.

## 결론

실험 6의 2-stage classifier는 exp5의 병목을 해결하지 못했다. 단순한 문제 분해만으로는 BASK와 non-BASK의 경계가 충분히 분리되지 않았으며, 오히려 Stage 1이 병목이 되었다.

다음 개선은 2-stage 구조를 유지하기보다 다음 방향이 더 타당하다.

1. Stage 1을 `BASK detector`가 아니라 `BFSK/BPSK 보호형` decision으로 재설계한다.
2. threshold objective를 BASK recall 하한 중심이 아니라 macro F1 또는 worst recall 중심으로 바꾼다.
3. BASK/BFSK/BPSK의 시간 영역 window만 쓰지 말고, offset 근처 spectral shape 또는 cyclostationary-like feature를 추가한다.
4. 복원 목적까지 고려하면 classifier confidence가 낮은 샘플은 단일 예측으로 확정하지 않고, top-2 demod 후보를 모두 실행해 CRC로 선택하는 방식이 더 현실적이다.

## 산출물

- `results/stage1_bask_vs_nonbask/checkpoints/best.pt`
- `results/stage2_bfsk_vs_bpsk/checkpoints/best.pt`
- `results/two_stage_eval/two_stage_summary.md`
- `results/two_stage_eval/test_metrics.json`
- `results/two_stage_eval/test_stage_predictions.csv`
- `results/two_stage_eval/test_confusion_matrix.png`
- `results/analysis/exp06_two_stage_error_analysis.md`
