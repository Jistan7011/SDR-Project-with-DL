# Experiment 7: Unknown-Protocol Absorption Guard

## 1. 환경 확인

```powershell
cd D:\ai_projects\SDR\experiments\exp7\sourcecode
..\..\..\.venv\Scripts\python.exe -m pytest -q ..\tests
```

## 2. exp5 processed dataset 참조 import

```powershell
..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp07_unknown_protocol_windows --config ..\config\config.exp07.yaml --source-root ..\..\exp5\data\processed --output-root ..\data\processed
```

## 3. Evidence 추출

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.extract_exp07_signal_evidence --config ..\config\config.exp07.yaml --checkpoint ..\..\exp5\results\exp05_resnet_2048\resnet1d_seed42\checkpoints\best.pt --data-root ..\data\processed --output-root ..\data\evidence
```

## 4. Smoke 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_exp07_absorption_guard --config ..\config\config.exp07.yaml --evidence-root ..\data\evidence --output-dir ..\results\smoke\absorption_guard --preset smoke --seed 42
```

## 5. 공식 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_exp07_absorption_guard --config ..\config\config.exp07.yaml --evidence-root ..\data\evidence --output-dir ..\results\absorption_guard --preset train --seed 42
```

## 6. Unknown-protocol 평가

```powershell
..\..\..\.venv\Scripts\python.exe -m src.app.evaluate_exp07_unknown_analysis --config ..\config\config.exp07.yaml --checkpoint ..\..\exp5\results\exp05_resnet_2048\resnet1d_seed42\checkpoints\best.pt --guard-checkpoint ..\results\absorption_guard\checkpoints\best.pt --data-root ..\data\processed --output-dir ..\results\unknown_analysis
```

## 7. 결과 분석

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp07_results --config ..\config\config.exp07.yaml --summary ..\results\unknown_analysis\exp07_unknown_analysis_summary.json --output-dir ..\results\analysis
```

## 8. 신규 OTA session 수집

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.experiment.run_exp07_capture_session --session-id session_031 --config ..\config\config.exp07.yaml --dry-run"
```

Dry-run 확인 후 `--dry-run`을 제거하고 `session_031 ~ session_060`을 수집한다.
