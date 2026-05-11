# Milestone1 Fixed OTA Model Comparison

## Purpose

Milestone1 is the integrated fixed OTA experiment.

It compares O'Shea-style raw IQ models with Exp5/8/9-style models on the same fixed OTA dataset:

| Group | Model |
| --- | --- |
| O'Shea raw IQ | `oshea2018_resnet1d`, `oshea2018_vgg1d` |
| Exp5 style | `resnet1d_5ch` |
| Exp8 style | `multitask_resnet1d_5ch`, `multitask_resnet1d_margin_5ch` |
| Exp9 style | `rf_preprocessed_resnet1d` |

The pushed repository does not include generated SDR captures, processed datasets, checkpoints, or full result folders. A user who pulls the repository can reproduce them by following the stages below.

## Validated State

Current local validation:

- Python tests: `15 passed`
- PowerShell script parse check: all `run_fixed_*.ps1` scripts parse successfully
- Stage0/Preflight/Collection/Stage1/Stage1.5 were completed locally before artifact cleanup
- Stage2 is planned for full balanced fixed training

## Directory

```text
experiments/milestone1/
  config/config.milestone1.fixed.yaml
  sourcecode/src/
  tests/
  run_fixed_stage0_check.ps1
  run_fixed_preflight.ps1
  run_fixed_collect.ps1
  run_fixed_process_stage1.ps1
  run_fixed_stage2_if_pass.ps1
  PLAN.md
  STAGE1_STAGE15_REPORT.md
  README.md
  data/       # generated locally, ignored by git
  results/    # generated locally, ignored by git
```

## Environment

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For SDR collection, install radioconda with SoapySDR, HackRF, RTL-SDR support, then set:

```powershell
$env:RADIOCONDA_ROOT = "C:\Users\<you>\radioconda"
```

The scripts prefer the repository-local `.venv\Scripts\python.exe` and fall back to `python` if it does not exist.

## Verify Installation

```powershell
cd D:\ai_projects\SDR\experiments\milestone1\sourcecode
$env:PYTHONPATH = (Get-Location).Path
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pytest -q ..\tests
```

Expected in the current validated state:

```text
15 passed
```

Dry-run the capture command graph without SDR I/O:

```powershell
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$env:PYTHONPATH = (Get-Location).Path

& $PY -m src.experiment.run_oshea2018_capture_session `
  --config ..\config\config.milestone1.fixed.yaml `
  --session-id session_999 `
  --output-root ..\data\dry_run_check `
  --captures-per-class 1 `
  --dry-run `
  --tx-backend hackrf_transfer
```

## Stage Flow

| Stage | Purpose | Main Output |
| --- | --- | --- |
| Stage0 | Verify HackRF TX, RTL-SDR RX, simultaneous RX/TX capture | `data/hardware_check` |
| Preflight | Scan RF gain/offset conditions and select usable BASK/BPSK conditions | `results/fixed_preflight_analysis` |
| Fixed collection | Collect 20 fixed OTA sessions | `data/raw_ota_fixed` |
| Stage1 | Process fixed data and run quick 6-model comparison | `results/fixed_stage1_summary.*` |
| Stage1.5 | Train all 6 models on 25 percent of Stage2 data | `results/fixed_stage15_q25_*` |
| Stage2 | Train all 6 models on full balanced fixed dataset | `results/fixed_full_*` |
| Stage3 | Summarize final comparison and choose best model | final report |

## Data Policy

Payload bits are randomized.

- Every capture uses random payload bits.
- For the same `session_id + capture_idx`, BASK/BFSK/BPSK share the same `payload_seed`.
- Different session/capture pairs use different seeds.
- This prevents the model from using payload pattern as a shortcut.

## Run From Scratch

Run these commands from `experiments/milestone1`.

### 1. Stage0 Hardware Check

```powershell
cd D:\ai_projects\SDR\experiments\milestone1
$env:RADIOCONDA_ROOT = "C:\Users\<you>\radioconda"
.\run_fixed_stage0_check.ps1
```

This must finish:

- SoapySDR device diagnose
- RTL-SDR noise-only capture
- HackRF standalone BASK/BFSK/BPSK TX
- simultaneous RX/TX one-capture session

### 2. Preflight

```powershell
.\run_fixed_preflight.ps1
```

It scans:

- `tx_vga = 25, 30, 35`
- `rx_gain = 25, 30, 35`
- `offset = 250000, 500000`

Gate:

- BASK pass rate >= 80 percent
- BPSK pass rate >= 80 percent
- balanced pass rate >= 80 percent

### 3. Fixed Collection

```powershell
.\run_fixed_collect.ps1
```

Default collection:

- 20 sessions
- per session: `noise_only + BASK/BFSK/BPSK each 10 captures`
- only preflight-approved RF conditions are used

### 4. Process Data and Run Stage1

```powershell
.\run_fixed_process_stage1.ps1
```

This creates:

- `data/ota_processed_fixed`
- `data/ota_processed_fixed_balanced`
- `data/ota_processed_fixed_balanced_quick`
- `data/ota_rf_preprocessed_fixed_balanced_quick`
- `results/fixed_quick_*`
- `results/fixed_stage1_summary.json`

Stage1 gate:

- `BPSK recall >= 0.30`
- `worst recall > 0`
- BPSK must not be entirely absorbed into BASK

### 5. Stage1.5 Q25

Stage1.5 is a reduced-data intermediate run using one quarter of Stage2 data. It was run locally as a manual command sequence. Recreate it with:

```powershell
cd D:\ai_projects\SDR\experiments\milestone1\sourcecode
$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.milestone1.fixed.yaml"
$BAL_Q25 = "..\data\ota_processed_fixed_balanced_stage15_q25"
$RF_Q25  = "..\data\ota_rf_preprocessed_fixed_balanced_stage15_q25"

& $PY -m src.dataset.make_balanced_oshea2018_subset `
  --source-root ..\data\ota_processed_fixed_balanced `
  --output-root $BAL_Q25 `
  --train-per-session-class 2250 `
  --val-per-session-class 3000 `
  --test-per-session-class 2500 `
  --train-min-per-session-class 1000 `
  --val-min-per-session-class 1000 `
  --test-min-per-session-class 1000

& $PY -m src.dataset.prepare_oshea2018_rf_preprocessed `
  --config $CFG `
  --source-root $BAL_Q25 `
  --output-root $RF_Q25
```

Then train/evaluate the same six model commands used in `run_fixed_stage2_if_pass.ps1`, replacing:

- `..\data\ota_processed_fixed_balanced` with `$BAL_Q25`
- `..\data\ota_rf_preprocessed_fixed_balanced` with `$RF_Q25`
- output prefix `fixed_full_` with `fixed_stage15_q25_`

Stage1.5 local result summary is documented in `STAGE1_STAGE15_REPORT.md`.

### 6. Stage2 Full Training

```powershell
cd D:\ai_projects\SDR\experiments\milestone1
.\run_fixed_stage2_if_pass.ps1
```

Milestone1 intentionally trains all six models in Stage2. Raw ResNet/VGG failed quick gate in one Stage1 run but recovered in Stage1.5, so they remain part of the full comparison.

## Expected Local Dataset Sizes

These are generated locally and ignored by git:

| Dataset | Windows |
| --- | ---: |
| `ota_processed_fixed` | 1,309,344 |
| `ota_processed_fixed_balanced` | 540,000 |
| `ota_processed_fixed_balanced_quick` | 17,100 |
| `ota_processed_fixed_balanced_stage15_q25` | 135,000 |

## GitHub Policy

Do not commit generated artifacts:

- `data/**`
- `results/**`
- `*.bin`
- `*.npz`
- `*.pt`
- `*.pkl`

GitHub should contain source code, config, scripts, tests, and documentation only.
