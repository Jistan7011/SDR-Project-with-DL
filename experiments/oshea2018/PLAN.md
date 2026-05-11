# Oshea2018 원형 실험 계획

## Summary

`oshea2018`은 O'Shea 2018 논문식 raw IQ 기반 modulation classification 흐름을 재현하는 baseline 실험이다. 이 폴더는 논문 원형에 가까운 raw IQ 모델만 유지한다.

Fixed OTA 데이터에서 raw 모델과 Exp5/8/9 계열 모델을 비교하는 통합 실험은 `experiments/milestone1`로 분리했다.

## Scope

포함:

- synthetic BASK/BFSK/BPSK raw IQ 생성
- HOS/XGBoost baseline
- raw ResNet
- raw VGG
- raw IQ OTA baseline

제외:

- 5ch feature model
- MultiTask 5ch
- MultiTask margin 5ch
- RF-preprocessed ResNet
- fixed OTA Stage1/Stage1.5/Stage2/Stage3 비교

## Execution

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.yaml"
```

Synthetic dataset:

```powershell
& $PY -m src.dataset.generate_oshea2018_synthetic `
  --config $CFG `
  --preset train `
  --output-root ..\data\synthetic
```

Raw ResNet:

```powershell
& $PY -m src.train.train_oshea2018 `
  --config $CFG `
  --data-root ..\data\synthetic `
  --model-type oshea2018_resnet1d `
  --preset train `
  --output-dir ..\results\synthetic_resnet
```

Raw VGG:

```powershell
& $PY -m src.train.train_oshea2018 `
  --config $CFG `
  --data-root ..\data\synthetic `
  --model-type oshea2018_vgg1d `
  --preset train `
  --output-dir ..\results\synthetic_vgg
```

## Notes

- 이 폴더의 공식 입력은 raw I/Q `[2, 1024]`다.
- fixed OTA 비교 실험은 `milestone1`에서 진행한다.
- 기존 clean OTA 실패 분석 자료는 `milestone1/data/archive_clean_failure`와 `milestone1/results/archive_clean_failure`에 보존한다.
