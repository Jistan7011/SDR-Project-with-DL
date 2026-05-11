# Experiment 5 Plan

실험 5는 exp4의 핵심 병목인 `BFSK/BPSK -> BASK` 오분류를 줄이기 위한 분류기 개선 실험이다. exp4 raw IQ 15 session은 read-only reference로 사용하고, exp5에서 신규 1 m OTA 15 session을 추가 수집해 총 30 session 규모의 session-held-out dataset을 구성한다.

## 변경 사항

- 새 루트: `experiments/exp5/`
- 기본 window size: `2048`
- 보조 window size: `4096`
- 기본 feature: `[I,Q,magnitude,instantaneous_frequency,differential_phase]`
- 비교 모델: `ResNet1D`, `FusionResNet1D`
- 선택 기준: `balanced_score`, `worst_recall`
- calibration: validation split에서 class logit bias 탐색

## Split

- train: exp4 `session_001~012` + exp5 `session_016~024`
- val: exp4 `session_013~015` + exp5 `session_025~026`
- test: exp5 `session_027~030`

## 성공 기준

- minimum: accuracy >= 0.78, BFSK recall >= 0.75, BPSK recall >= 0.75, worst recall >= 0.74
- target: accuracy >= 0.82, macro F1 >= 0.82, BFSK/BPSK -> BASK error rate 35% 이상 감소
