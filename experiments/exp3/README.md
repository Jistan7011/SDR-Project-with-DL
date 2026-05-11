# Experiment 3

실험 3은 `10 cm OTA 안테나 송수신` 조건에서 확보한 SDR 변조 분류 모델이 `1 m OTA 안테나 송수신` 조건에서도 BASK/BFSK/BPSK를 분류할 수 있는지 검증한 실험이다.

공식 정리본은 다음 파일 하나를 우선해서 읽는다.

```text
experiments/exp3/FINAL_REPORT.md
```

## 현재 결론

```text
1 m OTA distance generalization:
  성공

Feature Fusion replacement:
  현재 구현 기준 실패

accuracy 기준 best:
  full fine-tune seed44
  accuracy = 0.7704

class balance 기준 후보:
  classifier-only fine-tune seed44
  worst recall = 0.7431
```

## 공식 Checkpoint

accuracy 기준:

```text
experiments/exp3/results/exp03_domain_adapt_ifreq_finetune_seed44/resnet1d_seed44/checkpoints/best.pt
```

class balance 기준:

```text
experiments/exp3/results/exp03_domain_adapt_ifreq_finetune_classifier_seed44/resnet1d_seed44/checkpoints/best.pt
```

## 폴더 구조

```text
experiments/exp3/
  FINAL_REPORT.md          # 공식 통합 보고서
  README.md                # 현재 안내 파일
  PLAN.md                  # 최초 계획서
  config/                  # 실험별 config
  sourcecode/              # exp3 전용 코드
  tests/                   # exp3 테스트
  data/raw_iq/             # 1 m OTA raw capture
  data/processed/          # strict exp3 processed dataset
  data/processed_domain_adapt/ # 1 m calibration 포함 dataset
  results/                 # checkpoint, logs, reports
```

## 재현 기본 위치

```powershell
cd D:\ai_projects\SDR\experiments\exp3\sourcecode
```

핵심 명령과 전체 과정은 `FINAL_REPORT.md`의 `5. 실행 과정`을 따른다.

## 테스트

```powershell
cd D:\ai_projects\SDR
.\.venv\Scripts\python.exe -m pytest -q experiments\exp3\tests
```

검증 결과:

```text
13 passed
```
