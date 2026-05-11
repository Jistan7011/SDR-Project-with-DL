# Experiment 6: 2-Stage Classifier

실험 6은 exp5의 병목인 `BFSK/BPSK -> BASK` 오분류를 줄이기 위해 2-stage classifier를 검증한다.

## 1. 환경 확인

```powershell
cd D:\ai_projects\SDR\experiments\exp6\sourcecode

..\..\..\.venv\Scripts\python.exe -m pytest -q ..\tests
```

## 2. Binary dataset 생성

```powershell
Remove-Item -Recurse -Force ..\data\processed_stage -ErrorAction SilentlyContinue

..\..\..\.venv\Scripts\python.exe -m src.dataset.make_exp06_binary_datasets --config ..\config\config.exp06.yaml
```

## 3. Smoke 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_binary_classifier --config ..\config\config.exp06.yaml --stage stage1 --data-root ..\data\processed_stage\bask_vs_nonbask --output-dir ..\results\smoke\stage1 --preset smoke --seed 42

..\..\..\.venv\Scripts\python.exe -m src.train.train_binary_classifier --config ..\config\config.exp06.yaml --stage stage2 --data-root ..\data\processed_stage\bfsk_vs_bpsk --output-dir ..\results\smoke\stage2 --preset smoke --seed 42
```

## 4. 본 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_binary_classifier --config ..\config\config.exp06.yaml --stage stage1 --data-root ..\data\processed_stage\bask_vs_nonbask --output-dir ..\results\stage1_bask_vs_nonbask --preset train --seed 42

..\..\..\.venv\Scripts\python.exe -m src.train.train_binary_classifier --config ..\config\config.exp06.yaml --stage stage2 --data-root ..\data\processed_stage\bfsk_vs_bpsk --output-dir ..\results\stage2_bfsk_vs_bpsk --preset train --seed 42
```

## 5. Two-stage 평가

```powershell
..\..\..\.venv\Scripts\python.exe -m src.app.evaluate_two_stage_classifier --config ..\config\config.exp06.yaml --stage1-checkpoint ..\results\stage1_bask_vs_nonbask\checkpoints\best.pt --stage2-checkpoint ..\results\stage2_bfsk_vs_bpsk\checkpoints\best.pt --data-root ..\..\exp5\data\processed --output-dir ..\results\two_stage_eval
```

## 6. 오류 분석

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp06_two_stage_errors --summary ..\results\two_stage_eval\two_stage_summary.json --output-dir ..\results\analysis
```

## 산출물

- `data/processed_stage/manifest_exp06_binary.json`
- `results/stage1_bask_vs_nonbask/checkpoints/best.pt`
- `results/stage2_bfsk_vs_bpsk/checkpoints/best.pt`
- `results/two_stage_eval/two_stage_summary.md`
- `results/analysis/exp06_two_stage_error_analysis.md`
