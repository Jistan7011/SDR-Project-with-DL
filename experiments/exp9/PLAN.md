# 실험 9 공식 계획: RF Canonicalization + Feature Fusion 기반 Blind 변조 분류 개선

## Summary

- 목적은 blind `BASK/BFSK/BPSK` 분류에서 남은 `BFSK/BPSK -> BASK` 흡수 오류를 RF 전처리와 feature fusion으로 줄이는 것이다.
- 통신 규약 미지 조건을 유지하며 `preamble/sync/CRC/payload`는 모델 입력으로 사용하지 않는다.
- 1차 데이터는 exp8의 60-session processed dataset을 재사용한다.

## Implementation

- `prepare_exp09_preprocessed_dataset`: DC removal, RMS normalization, coarse CFO correction, phase slope metadata, spectral/evidence feature 저장.
- `analyze_exp09_preprocessing`: 전처리 전후 proxy metadata와 evidence score 분포 분석.
- `run_exp09_seed_sweep`: `resnet1d`, `fusion_resnet1d_exp9`, `fusion_resnet1d_exp9_margin` seed sweep 실행.
- `evaluate_exp09_blind_classifier`: 동일 test split에서 accuracy, macro F1, worst recall, BASK absorption rate 평가.
- `analyze_exp09_results`: Exp8 기준과 비교 가능한 공식 결과표 작성.

## Success Criteria

- 최소 기준: accuracy `>= 0.76`, macro F1 `>= 0.76`, worst recall `>= 0.73`, BFSK recall `>= 0.75`, BPSK recall `>= 0.74`, `BFSK/BPSK -> BASK <= 0.09`.
- 목표 기준: accuracy `>= 0.80`, macro F1 `>= 0.80`, worst recall `>= 0.76`, `BFSK/BPSK -> BASK <= 0.06`.

## Notes

- 신규 SDR 수집은 실험 9 1차 후보 실패 후 boundary-targeted capture로 분리한다.
- 실험 9는 payload recovery가 아니라 blind modulation classification 개선 실험이다.
