# 실험 7 계획: BASK 흡수 방지형 Unknown-Protocol RF Signal Analysis

실험 7은 exp5/exp6에서 반복된 `BFSK/BPSK -> BASK` 흡수 문제를 줄이기 위해 top-1 label을 바로 확정하지 않고, classifier confidence와 DSP/statistical evidence를 함께 사용한다.

분석기는 CRC, preamble, sync word, payload pool을 입력으로 사용하지 않는다. 해당 정보는 ground-truth 평가용 metadata로만 유지한다.

## 공식 흐름

```text
exp5/exp7 processed IQ
-> base ResNet1D softmax
-> evidence feature extraction
-> absorption guard MLP
-> unknown-protocol final decision
-> hard-BASK error / candidate retention 평가
```

## 주요 산출물

- `data/evidence/*_evidence.npz`
- `results/absorption_guard/checkpoints/best.pt`
- `results/unknown_analysis/exp07_unknown_analysis_summary.md`
- `results/analysis/exp07_analysis.md`

## 성공 기준

- forced 3-class accuracy >= 0.74
- worst recall >= 0.72
- BFSK/BPSK hard-BASK final decision rate <= 0.12
- BFSK/BPSK candidate retention rate >= 0.90
- true BASK hard-BASK precision >= 0.70
- ambiguous decision rate <= 0.25
