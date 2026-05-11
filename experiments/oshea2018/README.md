# O'Shea 2018 Raw IQ Reproduction

## Purpose

This experiment keeps only the original O'Shea-style raw IQ baseline flow.

It is intentionally separated from `experiments/milestone1`.

- `oshea2018`: raw IQ reproduction baseline
- `milestone1`: fixed OTA dataset comparison between raw O'Shea models and Exp5/8/9-style models

The pushed repository does not include generated `.npz`, `.bin`, `.pt`, or model artifact files. They are regenerated locally by the commands below.

## Models

| Model | Input | Purpose |
| --- | --- | --- |
| `oshea2018_resnet1d` | raw I/Q `[2, 1024]` | O'Shea 2018-style residual raw IQ CNN |
| `oshea2018_vgg1d` | raw I/Q `[2, 1024]` | O'Shea 2018-style VGG raw IQ CNN |
| `hos_xgboost_baseline` | hand-crafted HOS features | Classical baseline for synthetic comparison |

5-channel features, multitask heads, margin loss, RF preprocessing, fixed OTA Stage1/Stage1.5/Stage2 are not part of this folder. Use `experiments/milestone1` for those.

## Directory

```text
experiments/oshea2018/
  config/config.oshea2018.yaml
  sourcecode/src/
  tests/
  data/        # generated locally, ignored by git
  results/     # generated locally, ignored by git
  PLAN.md
  README.md
```

## Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For SDR/OTA capture, install radioconda/SoapySDR/HackRF/RTL-SDR separately and set:

```powershell
$env:RADIOCONDA_ROOT = "C:\Users\<you>\radioconda"
```

Synthetic reproduction does not require SDR hardware.

## Verify Installation

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pytest -q ..\tests
```

Expected in the current validated state:

```text
12 passed
```

## Reproduce Synthetic Raw IQ Baseline

Run from `experiments/oshea2018/sourcecode`.

### 1. Generate Synthetic Dataset

Smoke dataset:

```powershell
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$env:PYTHONPATH = (Get-Location).Path

& $PY -m src.dataset.generate_oshea2018_synthetic `
  --config ..\config\config.oshea2018.yaml `
  --preset smoke `
  --output-root ..\data\synthetic_smoke
```

Full synthetic dataset:

```powershell
& $PY -m src.dataset.generate_oshea2018_synthetic `
  --config ..\config\config.oshea2018.yaml `
  --preset train `
  --output-root ..\data\synthetic
```

### 2. Train Raw ResNet

```powershell
& $PY -m src.train.train_oshea2018 `
  --config ..\config\config.oshea2018.yaml `
  --data-root ..\data\synthetic `
  --model-type oshea2018_resnet1d `
  --preset train `
  --output-dir ..\results\synthetic_resnet
```

### 3. Evaluate Raw ResNet

```powershell
& $PY -m src.app.evaluate_oshea2018 `
  --config ..\config\config.oshea2018.yaml `
  --checkpoint ..\results\synthetic_resnet\checkpoints\best.pt `
  --data-root ..\data\synthetic `
  --output-dir ..\results\synthetic_resnet
```

### 4. Train/Evaluate Raw VGG

```powershell
& $PY -m src.train.train_oshea2018 `
  --config ..\config\config.oshea2018.yaml `
  --data-root ..\data\synthetic `
  --model-type oshea2018_vgg1d `
  --preset train `
  --output-dir ..\results\synthetic_vgg

& $PY -m src.app.evaluate_oshea2018 `
  --config ..\config\config.oshea2018.yaml `
  --checkpoint ..\results\synthetic_vgg\checkpoints\best.pt `
  --data-root ..\data\synthetic `
  --output-dir ..\results\synthetic_vgg
```

## Optional OTA Raw IQ Flow

The config still contains raw OTA fields for paper-style OTA baseline work. OTA capture requires HackRF One TX and RTL-SDR RX.

Minimal dry-run:

```powershell
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config ..\config\config.oshea2018.yaml `
  --session-id session_001 `
  --output-root ..\data\raw_ota_clean `
  --captures-per-class 1 `
  --dry-run
```

Real OTA capture:

```powershell
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config ..\config\config.oshea2018.yaml `
  --session-id session_001 `
  --output-root ..\data\raw_ota_clean `
  --captures-per-class 10 `
  --radioconda-root $env:RADIOCONDA_ROOT
```

## GitHub Policy

Do not commit generated data or model artifacts:

- `data/**`
- `results/**`
- `*.bin`
- `*.npz`
- `*.pt`
- `*.pkl`

Only source code, config, tests, and documentation are intended for GitHub.
