# Experiment 8: Blind 3-Class Modulation Classifier Improvement

Experiment 8 improves blind `BASK/BFSK/BPSK` classification after Exp7. It does
not use payload, CRC, preamble, or sync as model input.

## 1. Test

```powershell
cd D:\ai_projects\SDR\experiments\exp8\sourcecode
..\..\..\.venv\Scripts\python.exe -m pytest -q ..\tests
```

## 2. Prepare Dataset

```powershell
..\..\..\.venv\Scripts\python.exe -m src.dataset.prepare_exp08_blind_dataset --source-root ..\..\exp7\data\processed_retrained --output-root ..\data\processed
```

Expected split size:

- train: 25200
- val: 5400
- test: 5400

## 3. Mine Hard Negatives

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.mine_exp08_hard_negatives --baseline-checkpoint ..\..\exp7\results\base_retrained_seed42\checkpoints\best.pt --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-dir ..\results\hard_negative_mining
```

## 4. Smoke Train

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_exp08_multitask_classifier --config ..\config\config.exp08.yaml --preset smoke --seed 42 --output-dir ..\results\smoke\multitask_seed42 --data-root ..\data\processed --model-type multitask_resnet1d

..\..\..\.venv\Scripts\python.exe -m src.app.evaluate_exp08_blind_classifier --checkpoint ..\results\smoke\multitask_seed42\checkpoints\best.pt --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-dir ..\results\smoke\multitask_seed42
```

## 5. Full Seed Sweep

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_exp08_seed_sweep --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-root ..\results\exp08_multitask_resnet --model-type multitask_resnet1d --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp08_seed_sweep --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-root ..\results\exp08_multitask_resnet_margin --model-type multitask_resnet1d_margin --seeds 42 43 44
```

## 6. Analyze

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp08_results --results-root ..\results --output-dir ..\results\analysis
```
