# Experiment 4: End-to-End OTA Modulation Classification + Data Recovery

This experiment checks whether the project can go beyond “which modulation is this IQ window?” and recover actual transmitted payload data.

RF path:

- HackRF One transmits.
- RTL-SDR Blog V4 receives.
- The SDRs are connected to the same PC by USB only.
- The RF signal travels over the air through antennas.
- Default distance is 1 m, face-to-face antenna layout.

## 1. Environment Check

Run from the experiment source directory:

```powershell
cd D:\ai_projects\SDR\experiments\exp4\sourcecode

..\..\..\.venv\Scripts\python.exe --version

..\..\..\.venv\Scripts\python.exe -m pytest -q ..\tests

cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

## 2. Dry-Run the SDR Commands

This prints the RX/TX commands without transmitting.

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp04_recovery_session --session-id session_001 --distance-m 1.0 --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000 --dry-run
```

## 3. Capture One Recovery Session

Place the HackRF and RTL-SDR antennas about 1 m apart, facing each other.

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp04_recovery_session --session-id session_001 --distance-m 1.0 --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000
```

For the official Experiment 4 run, use the full 15-session capture plan instead of a single session:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp04_recovery_plan --config ..\config\config.exp04.yaml
```

The full plan captures:

```text
15 sessions
30 captures per session = BASK/BFSK/BPSK x 10 payloads
450 total OTA captures
```

Expected raw data:

```text
experiments\exp4\data\raw_iq\session_001\noise_only.bin
experiments\exp4\data\raw_iq\session_001\<modulation>_<payload>_offset500000_txvga30.bin
experiments\exp4\data\raw_iq\session_001\metadata.json
```

## 4. Import Frame-Level Recovery Samples

```powershell
..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp04_recovery_frames --raw-root ..\data\raw_iq --output-root ..\data\processed
```

This channelizes the configured offset to DC, downsamples to 160 kS/s, preserves the longer active IQ burst as `raw_iq`, and stores classifier windows as `iq`.

Default imported sample count:

```text
20 frame windows per capture
train: 5400 samples
val: 1800 samples
test: 1800 samples
total: 9000 samples
```

The split is session-held-out:

```text
train: session_001 ~ session_009
val: session_010 ~ session_012
test: session_013 ~ session_015
```

## 5. Optional Fine-Tuning on Experiment 4 Data

Experiment 4 can either evaluate the Experiment 3 checkpoint directly or fine-tune it with the larger Experiment 4 training split.

Fine-tune:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_cnn1d --config ..\config\config.exp04.yaml --preset train --model-type resnet1d --output-dir ..\results\exp04_resnet_finetune
```

Evaluate modulation classification on the held-out recovery test sessions:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint ..\results\exp04_resnet_finetune\checkpoints\best.pt --config ..\config\config.exp04.yaml --data-root ..\data\processed --split test --output-dir ..\results\exp04_resnet_finetune
```

## 6. Evaluate End-to-End Recovery

Default Experiment 3 classifier:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.app.evaluate_recovery --input-root ..\data\processed --checkpoint ..\..\exp3\results\exp03_domain_adapt_ifreq_finetune_seed44\resnet1d_seed44\checkpoints\best.pt --config ..\config\config.exp04.yaml --output-dir ..\results\reports\recovery_eval
```

Fine-tuned Experiment 4 classifier:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.app.evaluate_recovery --input-root ..\data\processed --checkpoint ..\results\exp04_resnet_finetune\checkpoints\best.pt --config ..\config\config.exp04.yaml --output-dir ..\results\reports\recovery_eval_finetuned
```

Generate grouped analysis and figures:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp04_recovery --results ..\results\reports\recovery_eval\recovery_eval.json --output-dir ..\results\reports
```

## 7. Outputs

Main files:

- `results/reports/recovery_eval/recovery_eval.json`
- `results/reports/recovery_eval/recovery_eval.csv`
- `results/reports/recovery_metrics.json`
- `results/reports/recovery_by_modulation.csv`
- `results/reports/recovery_by_payload.csv`
- `results/reports/recovery_by_session.csv`
- `results/reports/confusion_matrix.png`
- `results/reports/crc_packet_success.png`

The evaluation records classification confidence, margin, predicted modulation, expected modulation, recovered payload, CRC result, BER/CER, packet success, oracle-demod result, failure stage, and RF condition metadata.
