# Milestone1 실험 계획

## Summary

`milestone1`은 fixed OTA 데이터셋에서 O'Shea 2018 raw IQ 모델과 Exp5/8/9 계열 AI 모델을 공정 비교하는 통합 실험이다.

기존 `oshea2018`은 논문식 raw IQ baseline 재현 전용으로 복원하고, fixed OTA 비교 실험은 이 폴더에서 진행한다.

## Models

| 모델 | 입력 | 목적 |
| --- | --- | --- |
| raw ResNet | raw I/Q `[2, 1024]` | O'Shea 2018 raw IQ ResNet baseline |
| raw VGG | raw I/Q `[2, 1024]` | O'Shea 2018 raw IQ VGG baseline |
| 5ch ResNet | I/Q/magnitude/instantaneous frequency/differential phase | Exp5 계열 feature 확장 |
| MultiTask 5ch | 5ch + auxiliary heads | Exp8 계열 multitask 구조 |
| MultiTask margin 5ch | 5ch + multitask + contrastive margin | Exp8 개선 구조 |
| RF-preprocessed ResNet | RF-preprocessed 5ch | Exp9 계열 RF 전처리 비교 |

## Stage Plan

| Stage | 상태 | 목적 | 산출물 |
| --- | --- | --- | --- |
| Stage0 | 완료 | SDR TX/RX 체인 검증 | `data/hardware_check` |
| Preflight | 완료 | BASK/BPSK가 학습 가능한 RF 조건 선별 | `results/fixed_preflight_analysis` |
| Fixed Collection | 완료 | 20-session fixed OTA 데이터 수집 | `data/raw_ota_fixed` |
| Stage1 | 완료 | quick subset으로 BPSK collapse 및 gate 확인 | `results/fixed_stage1_summary.*` |
| Stage1.5 | 완료 | Stage2 1/4 데이터로 6개 모델 중간 비교 | `results/fixed_stage15_q25_*` |
| Stage2 | 미완료 | full balanced fixed dataset으로 6개 모델 학습 | `results/fixed_full_*`, `results/fixed_full_comparison` |
| Stage3 | 미완료 | 최종 분석, 모델 선정, 보고서 작성 | final comparison report |

## Data Policy

공식 비교는 fixed OTA dataset만 사용한다.

```text
raw:       data/raw_ota_fixed
processed: data/ota_processed_fixed
balanced:  data/ota_processed_fixed_balanced
stage1.5:  data/ota_processed_fixed_balanced_stage15_q25
```

기존 clean OTA 실패 자료는 삭제하지 않고 archive로 보존한다.

```text
data/archive_clean_failure
results/archive_clean_failure
```

## Stage2 Command

```powershell
cd D:\ai_projects\SDR\experiments\milestone1

New-Item -ItemType Directory -Force -Path .\results | Out-Null
$log = ".\results\fixed_stage2_all_models_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
.\run_fixed_stage2_if_pass.ps1 2>&1 | Tee-Object -FilePath $log
```

Stage2는 full balanced fixed dataset 540,000 windows를 사용한다. Stage1에서 quick gate를 실패한 raw ResNet/raw VGG도 Stage1.5에서 회복했으므로 full 비교 대상에 포함한다.

## Stage3 Definition

Stage3는 Stage2 이후 최종 분석 단계다.

- Stage2 full comparison 결과 정리
- Stage1.5와 Stage2의 성능 변화 비교
- best model 선정
- confusion matrix, worst recall, absorption, session별 accuracy 분석
- clean failure archive와 fixed official result 분리 보고
- 최종 보고서 작성

Best model 선정 기준:

1. worst recall
2. macro F1
3. absorption 낮음

## Notes

- 속도 병목은 `.npz` 다량 랜덤 로딩과 HDD I/O에서 주로 발생했다.
- 이후 대규모 반복 실험은 NVMe dataset 운용 또는 packed/sharded dataset을 권장한다.
- 상세 Stage1/Stage1.5 결과와 모델 구조 설명은 `STAGE1_STAGE15_REPORT.md`를 기준으로 한다.
