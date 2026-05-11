# Experiment 7 Final Report

## 요약

실험 7은 exp5/exp6에서 반복된 `BFSK/BPSK -> BASK` 흡수 오류를 줄이기 위해 unknown-protocol RF signal analysis 구조를 구현하고 개선했다.

분석기는 CRC, preamble, sync word, payload를 입력으로 사용하지 않는다. 이 정보들은 평가용 ground truth로만 보관한다. 목표는 실제 통신 규약을 모르는 상황에서 단일 hard modulation label을 무리하게 확정하지 않고, classifier 출력과 signal evidence를 함께 사용해 hard decision, ambiguous decision, unknown decision을 분리하는 것이다.

## 개선 전후 핵심 변화

초기 exp7은 hard-BASK 흡수율을 0까지 낮췄지만 `UNKNOWN_LOW_CONFIDENCE`가 너무 많아 ambiguous/unknown rate가 0.3837까지 올라갔다. 개선판에서는 다음을 수정했다.

- low-confidence sample도 top-k candidate를 보존하도록 변경
- `UNKNOWN_LOW_CONFIDENCE`를 forced metric에서 무조건 BASK로 환산하지 않도록 변경
- base classifier top1이 BFSK/BPSK인 저신뢰 sample은 hard non-BASK decision으로 유지
- BASK top1 저신뢰 sample은 여전히 unknown/ambiguous로 남겨 BASK 흡수를 방지
- threshold를 완화해 unknown 과다 발생을 줄임

## 데이터

공식 개선 평가는 exp5 processed dataset을 재사용했다.

- train: 12,600 samples
- val: 3,000 samples
- test: 2,400 samples
- split: exp5 session-held-out split 유지
- test sessions: `session_027`, `session_028`, `session_029`, `session_030`

신규 `session_031 ~ session_060` OTA 수집 CLI는 구현되어 있지만, 이 보고서의 공식 수치에는 아직 포함하지 않았다.

## 공식 개선 결과

결과 경로:

- `results/unknown_analysis_v4/`
- `results/analysis_v4/`

| metric | initial exp7 | improved exp7 v4 | minimum criterion | pass |
| --- | ---: | ---: | ---: | --- |
| forced 3-class accuracy | 0.7221 | 0.7304 | >= 0.74 | fail |
| macro F1 | 0.7265 | 0.7309 | - | - |
| worst-class recall | 0.5925 | 0.7050 | >= 0.72 | fail |
| BFSK/BPSK hard-BASK final decision rate | 0.0000 | 0.0000 | <= 0.12 | pass |
| BFSK/BPSK candidate retention rate | 0.8531 | 0.9681 | >= 0.90 | pass |
| true BASK hard-BASK precision | 1.0000 | 1.0000 | >= 0.70 | pass |
| ambiguous decision rate | 0.3837 | 0.1875 | <= 0.25 | pass |

Class recall:

| class | recall |
| --- | ---: |
| BASK | 0.7275 |
| BFSK | 0.7588 |
| BPSK | 0.7050 |

Forced 3-class confusion matrix:

| expected \ predicted | BASK | BFSK | BPSK |
| --- | ---: | ---: | ---: |
| BASK | 582 | 133 | 85 |
| BFSK | 132 | 607 | 61 |
| BPSK | 132 | 104 | 564 |

Final decision counts:

| decision | count |
| --- | ---: |
| BASK | 467 |
| BFSK | 773 |
| BPSK | 710 |
| AMBIGUOUS_BASK_LIKE_WITH_BFSK_EVIDENCE | 71 |
| UNKNOWN_LOW_CONFIDENCE | 379 |

## Exp5 Baseline 비교

| metric | exp5 baseline | improved exp7 v4 |
| --- | ---: | ---: |
| accuracy / forced accuracy | 0.7308 | 0.7304 |
| worst recall | 0.7050 | 0.7050 |
| BFSK/BPSK -> BASK rate | 0.1894 | 0.0000 |

개선판 exp7은 exp5의 forced accuracy와 worst recall을 거의 유지하면서, 가장 중요한 `BFSK/BPSK -> hard-BASK` 흡수 오류를 0으로 줄였다. 또한 candidate retention은 0.9681로 목표 기준에 가까운 수준까지 올라갔다.

## 해석

실험 7 개선 결과의 의미는 다음과 같다.

- BASK 흡수 방지는 성공했다.
- 후보 유지율도 충분히 개선됐다.
- ambiguous/unknown 과다 문제도 기준 이하로 낮췄다.
- 다만 forced 3-class accuracy는 exp5 baseline과 거의 같고, 최소 기준 0.74에는 아직 부족하다.
- worst recall도 0.7050으로 exp5 수준까지 회복했지만 최소 기준 0.72에는 도달하지 못했다.

즉, 실험 7 개선판은 “안전한 unknown-protocol 분석기”로는 의미가 있다. 잘못된 hard-BASK 확정을 피하면서 후보를 유지하는 데 성공했다. 그러나 순수 분류 성능 자체를 끌어올리려면 신규 OTA session 추가 또는 base classifier 재학습이 필요하다.

## 다음 단계

다음 실험에서는 성능 기준 통과를 위해 다음을 우선한다.

1. `session_031 ~ session_060` 신규 OTA 수집
   - exp5 30 session에서 60 session 규모로 확장
   - random 1-byte/2-byte payload 적용
   - 기존 고정 payload pool 의존 감소

2. base ResNet1D 재학습
   - exp7 evidence rule만으로는 forced accuracy가 exp5 baseline 근처에서 정체
   - 신규 session 포함 후 base classifier 자체를 다시 학습해야 함

3. threshold validation sweep 정식화
   - hard-BASK rate, candidate retention, ambiguous rate를 함께 최적화
   - 단일 threshold가 아니라 operating profile별 설정 보관

4. unknown-protocol report 강화
   - top-k candidate, evidence score, confidence/margin/entropy, condition metadata를 함께 남기는 분석 리포트 유지

## 검증

- exp7 unit/integration tests: `33 passed`
- official improved evaluation: `results/unknown_analysis_v4/`
- official improved analysis: `results/analysis_v4/`

## 신규 OTA Session 재학습 결과

추가 작업으로 `session_031 ~ session_060` 신규 1 m OTA 데이터를 수집하고, exp4/exp5/exp7 raw session을 합쳐 60-session dataset을 구성했다.

신규 수집 상태:

- 신규 sessions: `session_031` ~ `session_060`
- 각 session: noise-only 1개 + BASK/BFSK/BPSK x random payload 10개 = 31 binary IQ files
- 모든 신규 session에서 `metadata.json` 생성 확인
- 신규 payload: 고정 payload pool이 아니라 deterministic random 1-byte/2-byte payload 사용

재학습 dataset:

| split | sessions | samples |
| --- | ---: | ---: |
| train | 42 | 25,200 |
| val | 9 | 5,400 |
| test | 9 | 5,400 |
| total | 60 | 36,000 |

재학습 base classifier:

- checkpoint: `results/base_retrained_seed42/checkpoints/best.pt`
- test accuracy: 0.7243
- BASK recall: 0.7167
- BFSK recall: 0.7306
- BPSK recall: 0.7256
- worst recall: 0.7167

재학습 후 unknown-protocol analysis는 `high_frequency_separation=0.62` 운영점을 공식 선택했다.

| metric | retrained result |
| --- | ---: |
| forced 3-class accuracy | 0.7246 |
| macro F1 | 0.7253 |
| worst-class recall | 0.6356 |
| BFSK/BPSK hard-BASK final decision rate | 0.0108 |
| BFSK/BPSK candidate retention rate | 0.9022 |
| true BASK hard-BASK precision | 0.9647 |
| ambiguous decision rate | 0.1226 |

재학습 결과 해석:

- 신규 데이터 추가 후에도 forced accuracy는 0.7246으로 크게 오르지 않았다.
- base classifier의 class recall 균형은 좋아졌지만, absorption guard/evidence rule 적용 후 BASK recall이 낮아지는 문제가 남았다.
- 대신 unknown-protocol 안전 지표는 유지됐다. `BFSK/BPSK -> hard-BASK`는 0.0108까지 낮고, candidate retention은 0.9022로 최소 기준을 통과했다.
- 결론적으로 신규 session 재학습은 데이터 다양성 확대에는 성공했지만, 정확도 향상은 제한적이었다. 다음 개선은 단순 데이터 추가보다 BASK와 non-BASK boundary를 따로 보존하는 loss/objective 또는 stage-aware guard calibration이 필요하다.

재학습 산출물:

- `data/processed_retrained/`
- `data/evidence_retrained/`
- `results/base_retrained_seed42/`
- `results/absorption_guard_retrained_seed42/`
- `results/unknown_analysis_retrained_seed42_calibrated_hf062/`
- `results/analysis_retrained_seed42_hf062/`
