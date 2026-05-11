# SDR Project With Deep Learning

This repository contains staged SDR experiments for BASK, BFSK, and BPSK signal classification and recovery.

Typical flow:

```text
HackRF One TX
-> OTA RF channel
-> RTL-SDR Blog V4 RX
-> IQ capture / channelize / windowing
-> deep learning modulation classifier
-> optional DSP payload recovery
-> accuracy, recall, BER/CER, packet success analysis
```

## Repository Layout

```text
SDR/
  experiments/
    exp1/ ... exp9/
    oshea2018/
    milestone1/
    _template/
  reference/
  scripts/
  README_ML_DL_ENV.md
  PROJECT_STRUCTURE.md
  requirements.txt
```

Each experiment is self-contained:

```text
experiments/<name>/
  README.md
  PLAN.md
  config/
  sourcecode/
  tests/
  data/       # generated locally, ignored by git
  results/    # generated locally, ignored by git
```

## Main Experiments

| Folder | Meaning |
| --- | --- |
| `experiments/oshea2018` | O'Shea 2018-style raw IQ baseline reproduction |
| `experiments/milestone1` | Fixed OTA comparison of raw O'Shea models and Exp5/8/9-style models |
| `experiments/exp1` ... `experiments/exp9` | Earlier staged SDR experiments |

See `PROJECT_STRUCTURE.md` for the migration summary and GitHub push notes.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For hardware SDR experiments, install radioconda/SoapySDR/HackRF/RTL-SDR support and set:

```powershell
$env:RADIOCONDA_ROOT = "C:\Users\<you>\radioconda"
```

## Quick Validation

O'Shea raw IQ experiment:

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pytest -q ..\tests
```

Milestone1:

```powershell
cd D:\ai_projects\SDR\experiments\milestone1\sourcecode
$env:PYTHONPATH = (Get-Location).Path
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pytest -q ..\tests
```

Validated local state:

```text
oshea2018: 12 passed
milestone1: 15 passed
```

## GitHub Artifact Policy

Generated datasets and model artifacts are not committed.

Ignored examples:

- `data/**`
- `results/**`
- `*.bin`
- `*.npz`
- `*.npy`
- `*.pt`
- `*.pkl`

The repository should contain code, configs, scripts, tests, and documentation. A user should regenerate datasets and results by following each experiment's README.
