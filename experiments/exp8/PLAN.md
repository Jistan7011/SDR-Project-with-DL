# Experiment 8 Plan: Blind 3-Class Modulation Classifier Improvement

Experiment 8 targets the remaining blind classification bottleneck after Exp7: pure
`BASK/BFSK/BPSK` classification remains around the low 0.72 range even though the
unknown-analysis guard can prevent many hard BASK absorption decisions.

The experiment keeps the protocol-blind constraint. CRC, preamble, sync, and payload
are not model inputs. The primary change is a multi-task ResNet that learns the
3-class decision, the `BASK vs non-BASK` boundary, and the `BFSK vs BPSK` boundary
with a masked auxiliary head. Hard-negative mining uses only train split errors.

Minimum success:

- overall accuracy >= 0.76
- worst recall >= 0.72
- BASK recall >= 0.72
- BFSK recall >= 0.75
- BPSK recall >= 0.73
- BFSK/BPSK -> BASK error rate <= 0.08

Default command order:

```powershell
cd D:\ai_projects\SDR\experiments\exp8\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.dataset.prepare_exp08_blind_dataset --source-root ..\..\exp7\data\processed_retrained --output-root ..\data\processed

..\..\..\.venv\Scripts\python.exe -m src.analysis.mine_exp08_hard_negatives --baseline-checkpoint ..\..\exp7\results\base_retrained_seed42\checkpoints\best.pt --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-dir ..\results\hard_negative_mining

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp08_seed_sweep --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-root ..\results\exp08_multitask_resnet --model-type multitask_resnet1d --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.train.run_exp08_seed_sweep --config ..\config\config.exp08.yaml --data-root ..\data\processed --output-root ..\results\exp08_multitask_resnet_margin --model-type multitask_resnet1d_margin --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp08_results --results-root ..\results --output-dir ..\results\analysis
```
