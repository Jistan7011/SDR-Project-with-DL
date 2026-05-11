$ErrorActionPreference = "Stop"

$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location (Join-Path $SCRIPT_ROOT "sourcecode")
try {
$env:PYTHONPATH = (Get-Location).Path
$REPO_ROOT = (Resolve-Path (Join-Path $SCRIPT_ROOT "..\..")).Path
$PY = Join-Path $REPO_ROOT ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }
$CFG = "..\config\config.milestone1.fixed.yaml"
$RAW = "..\data\raw_ota_fixed"
$DATA = "..\data\ota_processed_fixed"
$BAL = "..\data\ota_processed_fixed_balanced"
$QUICK = "..\data\ota_processed_fixed_balanced_quick"
$RF_QUICK = "..\data\ota_rf_preprocessed_fixed_balanced_quick"

function Quote-CmdArg {
  param([Parameter(Mandatory=$true)][string]$Value)
  if ($Value -match '[\s&()^"]') {
    return '"' + ($Value -replace '"', '\"') + '"'
  }
  return $Value
}

function Invoke-PythonModule {
  param([Parameter(Mandatory=$true)][string[]]$Args)
  $parts = @((Quote-CmdArg $PY)) + ($Args | ForEach-Object { Quote-CmdArg $_ })
  $command = ($parts -join " ") + " 2>&1"
  cmd /c $command
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE`: $($Args -join ' ')"
  }
}

Remove-Item $DATA -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $BAL -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $QUICK -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $RF_QUICK -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ..\results\fixed_quick_* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ..\results\fixed_stage1_summary.* -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Import fixed raw OTA to processed windows ==="
Invoke-PythonModule @(
  "-m", "src.dataset.import_oshea2018_ota_windows",
  "--config", $CFG,
  "--raw-root", $RAW,
  "--output-root", $DATA,
  "--clean"
)

Write-Host ""
Write-Host "=== Create fixed balanced dataset ==="
Invoke-PythonModule @(
  "-m", "src.dataset.make_balanced_oshea2018_subset",
  "--source-root", $DATA,
  "--output-root", $BAL,
  "--train-per-session-class", "9000",
  "--val-per-session-class", "12000",
  "--test-per-session-class", "10000",
  "--train-min-per-session-class", "5000",
  "--val-min-per-session-class", "5000",
  "--test-min-per-session-class", "5000"
)

Write-Host ""
Write-Host "=== Create fixed quick balanced dataset ==="
Invoke-PythonModule @(
  "-m", "src.dataset.make_balanced_oshea2018_subset",
  "--source-root", $BAL,
  "--output-root", $QUICK,
  "--train-per-session-class", "300",
  "--val-per-session-class", "300",
  "--test-per-session-class", "300",
  "--train-min-per-session-class", "100",
  "--val-min-per-session-class", "100",
  "--test-min-per-session-class", "100"
)

Write-Host ""
Write-Host "=== Prepare RF-preprocessed quick dataset ==="
Invoke-PythonModule @(
  "-m", "src.dataset.prepare_oshea2018_rf_preprocessed",
  "--config", $CFG,
  "--source-root", $QUICK,
  "--output-root", $RF_QUICK
)

Write-Host ""
Write-Host "=== Stage1 quick: raw-IQ ResNet ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018", "--config", $CFG, "--data-root", $QUICK, "--model-type", "oshea2018_resnet1d", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_resnet")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_resnet\checkpoints\best.pt", "--data-root", $QUICK, "--output-dir", "..\results\fixed_quick_ota_resnet")
Write-Host "Completed: raw-IQ ResNet"

Write-Host ""
Write-Host "=== Stage1 quick: raw-IQ VGG ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018", "--config", $CFG, "--data-root", $QUICK, "--model-type", "oshea2018_vgg1d", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_vgg")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_vgg\checkpoints\best.pt", "--data-root", $QUICK, "--output-dir", "..\results\fixed_quick_ota_vgg")
Write-Host "Completed: raw-IQ VGG"

Write-Host ""
Write-Host "=== Stage1 quick: 5ch ResNet ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $QUICK, "--model-type", "resnet1d_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "cross_entropy", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_resnet_5ch")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_resnet_5ch\checkpoints\best.pt", "--data-root", $QUICK, "--output-dir", "..\results\fixed_quick_ota_resnet_5ch")
Write-Host "Completed: 5ch ResNet"

Write-Host ""
Write-Host "=== Stage1 quick: 5ch MultiTask ResNet ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $QUICK, "--model-type", "multitask_resnet1d_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "multitask", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_multitask_5ch")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_multitask_5ch\checkpoints\best.pt", "--data-root", $QUICK, "--output-dir", "..\results\fixed_quick_ota_multitask_5ch")
Write-Host "Completed: 5ch MultiTask ResNet"

Write-Host ""
Write-Host "=== Stage1 quick: 5ch MultiTask Margin ResNet ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $QUICK, "--model-type", "multitask_resnet1d_margin_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "multitask_margin", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_multitask_margin_5ch")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_multitask_margin_5ch\checkpoints\best.pt", "--data-root", $QUICK, "--output-dir", "..\results\fixed_quick_ota_multitask_margin_5ch")
Write-Host "Completed: 5ch MultiTask Margin ResNet"

Write-Host ""
Write-Host "=== Stage1 quick: RF-preprocessed ResNet ==="
Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $RF_QUICK, "--model-type", "rf_preprocessed_resnet1d", "--feature-mode", "iq", "--input-channels", "5", "--loss-type", "cross_entropy", "--preset", "smoke", "--output-dir", "..\results\fixed_quick_ota_rf_preprocessed_resnet")
Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_quick_ota_rf_preprocessed_resnet\checkpoints\best.pt", "--data-root", $RF_QUICK, "--output-dir", "..\results\fixed_quick_ota_rf_preprocessed_resnet")
Write-Host "Completed: RF-preprocessed ResNet"

Write-Host ""
Write-Host "=== Summarize Stage1 quick and gate Stage2 ==="
Invoke-PythonModule @(
  "-m", "src.analysis.summarize_oshea2018_stage1",
  "--results-root", "..\results",
  "--prefix", "fixed_quick_",
  "--output", "..\results\fixed_stage1_summary.json",
  "--min-bpsk-recall", "0.30"
)

Write-Host ""
Write-Host "Stage1 summary:"
Write-Host (Join-Path $SCRIPT_ROOT "results\fixed_stage1_summary.md")
Write-Host "If stage2_allowed is false, do not run full training; adjust RF collection first."
} finally {
  Pop-Location
}

