# 실험 9 최종 보고서

실험 9는 아직 실행 전이다. 구현 완료 후 `results/analysis/exp09_summary.md`와 평가 산출물을 기준으로 본 보고서를 갱신한다.

## 실행 순서

```powershell
cd D:\ai_projects\SDR\experiments\exp9\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.dataset.prepare_exp09_preprocessed_dataset --source-root ..\..\exp8\data\processed --output-root ..\data\processed --config ..\config\config.exp09.yaml

..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp09_preprocessing --data-root ..\data\processed --output-dir ..\results\preprocessing_analysis

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp09_seed_sweep --config ..\config\config.exp09.yaml --data-root ..\data\processed --output-root ..\results\exp09_resnet_preprocessed --model-type resnet1d --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp09_seed_sweep --config ..\config\config.exp09.yaml --data-root ..\data\processed --output-root ..\results\exp09_fusion_resnet --model-type fusion_resnet1d_exp9 --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp09_seed_sweep --config ..\config\config.exp09.yaml --data-root ..\data\processed --output-root ..\results\exp09_fusion_resnet_margin --model-type fusion_resnet1d_exp9_margin --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp09_results --results-root ..\results --output-dir ..\results\analysis
```
