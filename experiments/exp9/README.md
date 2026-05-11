# Experiment 9 - RF Canonicalization And Feature Fusion

## Purpose

Experiment 9 improves blind BASK/BFSK/BPSK classification by adding RF canonicalization and feature-fusion style inputs. It is a continuation of the earlier session-generalization experiments and focuses on reducing BFSK/BPSK absorption into BASK.

This is not a payload recovery experiment. It evaluates blind modulation classification only.

## Directory

```text
experiments/exp9/
  config/config.exp09.yaml
  sourcecode/
  tests/
  PLAN.md
  FINAL_REPORT.md
  data/       # generated locally, ignored by git
  results/    # generated locally, ignored by git
```

## Reproduce

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run tests from the experiment source folder:

```powershell
cd D:\ai_projects\SDR\experiments\exp9\sourcecode
$env:PYTHONPATH = (Get-Location).Path
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pytest -q ..\tests
```

Generated data and results are intentionally not committed. If a script requires an Exp8 processed dataset, regenerate or copy that dataset locally according to `PLAN.md` before running the Exp9 analysis/training commands.

## GitHub Policy

Do not commit:

- `data/**`
- `results/**`
- `*.bin`
- `*.npz`
- `*.pt`
- `*.pkl`

Commit only source code, configs, tests, and documentation.
