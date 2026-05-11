# Experiment 6 Plan

실험 6은 exp5에서 남은 `BFSK/BPSK -> BASK` 흡수 오류를 줄이기 위해 단일 3-class softmax를 2-stage classifier로 분해한다.

## 구조

- Stage 1: `BASK` vs `NON_BASK`
- Stage 2: `BFSK` vs `BPSK`
- 입력: exp5와 동일한 `[I,Q,magnitude,instantaneous_frequency,differential_phase]`
- window size: 2048
- 데이터: exp5 processed dataset 재사용

## 성공 기준

- accuracy >= 0.76
- BFSK recall >= 0.76
- BPSK recall >= 0.74
- worst recall >= 0.72
- BASK recall >= 0.72
- exp5 best의 BFSK/BPSK -> BASK rate `0.1894` 대비 25% 이상 감소
