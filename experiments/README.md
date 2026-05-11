# Experiments

Each experiment folder is self-contained. Code, config, tests, and documentation live with the experiment that produced them.

## Current Experiments

| Folder | Purpose |
| --- | --- |
| `exp1` | Initial real SDR baseline with random-window split |
| `exp2` | Session-held-out generalization |
| `exp3` | Distance/domain-shift generalization |
| `exp4` | End-to-end classification plus payload recovery |
| `exp5` | Improved OTA collection and learning method |
| `exp6` | Follow-up robustness/generalization experiment |
| `exp7` | Expanded session and payload variation experiment |
| `exp8` | Multitask 5-channel approach |
| `exp9` | RF canonicalization and feature-fusion approach |
| `oshea2018` | O'Shea 2018-style raw IQ reproduction |
| `milestone1` | Fixed OTA comparison across raw O'Shea and Exp5/8/9-style models |

## Standard Folder Shape

```text
experiments/<name>/
  README.md
  PLAN.md
  FINAL_REPORT.md       # when available
  config/
  sourcecode/
  tests/
  data/                 # generated locally, ignored by git
  results/              # generated locally, ignored by git
```

## Git Policy

Commit:

- README / PLAN / reports
- config files
- source code
- tests
- small metadata or manifest files only when intentionally documented

Do not commit generated large artifacts:

- raw IQ `.bin`
- processed dataset `.npz` / `.npy`
- checkpoints `.pt` / `.pth`
- serialized models `.pkl` / `.joblib`
- experiment `data/**`
- experiment `results/**`

Each experiment README should explain how to regenerate its local data/results.
