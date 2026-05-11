# SDR Repository Structure And GitHub Push Notes

## Current Structure

The repository has been reorganized around experiment folders.

```text
SDR/
  experiments/
    exp1/ ... exp9/
    oshea2018/
    milestone1/
    _template/
  reference/
  scripts/
  README.md
  README_ML_DL_ENV.md
  requirements.txt
```

The previous root-level `src/`, `tests/`, `data/`, and `results/` workflow has been removed in favor of experiment-local code snapshots.

## Important Experiment Folders

| Folder | Role | GitHub Contents |
| --- | --- | --- |
| `experiments/oshea2018` | O'Shea 2018 raw IQ reproduction | config, source, tests, README, PLAN |
| `experiments/milestone1` | Fixed OTA comparison across O'Shea raw models and Exp5/8/9 models | config, source, tests, scripts, README, PLAN, report |
| `experiments/exp1` ... `experiments/exp9` | Earlier staged experiments | source/config/docs only |

## Generated Files Are Local Only

Generated data and result artifacts are intentionally ignored:

- raw SDR captures: `*.bin`
- window datasets: `*.npz`, `*.npy`
- checkpoints: `*.pt`, `*.pth`
- serialized models: `*.pkl`, `*.joblib`, `*.onnx`
- experiment `data/**`
- experiment `results/**`

This keeps the repository below GitHub file size limits and forces reproducible regeneration from scripts.

## Validation Performed

Validated locally after the split:

```text
experiments/oshea2018: 12 tests passed
experiments/milestone1: 15 tests passed
milestone1 run_fixed_*.ps1: PowerShell parse check passed
oshea2018 synthetic smoke generate/train/evaluate: passed
milestone1 capture dry-run command graph: passed
```

Hardware-dependent Milestone1 Stage0/Preflight/Collection require HackRF One, RTL-SDR Blog V4, radioconda, and SoapySDR. They cannot be fully verified on a machine without the SDR devices.

## Before Pushing

Use these checks from repository root:

```powershell
git status --short
git check-ignore -v experiments\milestone1\data\raw_ota_fixed\session_001\session_001_bask_000.bin
git check-ignore -v experiments\oshea2018\data\synthetic\train\sample_0000000.npz
git check-ignore -v experiments\milestone1\results\fixed_stage15_q25_ota_resnet\checkpoints\best.pt
```

Check for accidental large tracked candidates:

```powershell
Get-ChildItem -Recurse -File -Force |
  Where-Object { $_.FullName -notmatch '\\.git\\|\\.venv\\' -and $_.Length -gt 90MB } |
  Select-Object FullName,Length
```

If any generated file appears in `git status`, do not commit it. Add or adjust `.gitignore` first.
