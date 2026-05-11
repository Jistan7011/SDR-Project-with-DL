$ErrorActionPreference = "Stop"

$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location (Join-Path $SCRIPT_ROOT "sourcecode")
try {
$env:PYTHONPATH = (Get-Location).Path
$REPO_ROOT = (Resolve-Path (Join-Path $SCRIPT_ROOT "..\..")).Path
$PY = Join-Path $REPO_ROOT ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }
$CFG = "..\config\config.milestone1.fixed.yaml"
$BAL = "..\data\ota_processed_fixed_balanced"
$RF_BAL = "..\data\ota_rf_preprocessed_fixed_balanced"
$SUMMARY = "..\results\fixed_stage1_summary.json"

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

if (!(Test-Path $SUMMARY)) {
  throw "Stage1 summary not found. Run run_fixed_process_stage1.ps1 first."
}

$summaryObj = Get-Content $SUMMARY | ConvertFrom-Json
if (-not $summaryObj.stage2_allowed) {
  Write-Host "Stage2 blocked: no Stage1 model reached the BPSK recall gate."
  Write-Host ("Check " + (Join-Path $SCRIPT_ROOT "results\fixed_stage1_summary.md"))
  exit 1
}

Remove-Item $RF_BAL -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ..\results\fixed_full_* -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Prepare RF-preprocessed full balanced dataset ==="
Invoke-PythonModule @("-m", "src.dataset.prepare_oshea2018_rf_preprocessed", "--config", $CFG, "--source-root", $BAL, "--output-root", $RF_BAL)

foreach ($row in $summaryObj.rows) {
  Write-Host ""
  Write-Host "=== Stage2 full training for $($row.model) ==="
  if (-not $row.stage2_pass) {
    Write-Host "Stage1 gate failed for this model, but full training is running by experiment override."
  }

  switch ($row.model) {
    "fixed_quick_ota_resnet" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018", "--config", $CFG, "--data-root", $BAL, "--model-type", "oshea2018_resnet1d", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_resnet")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_resnet\checkpoints\best.pt", "--data-root", $BAL, "--output-dir", "..\results\fixed_full_ota_resnet")
    }
    "fixed_quick_ota_vgg" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018", "--config", $CFG, "--data-root", $BAL, "--model-type", "oshea2018_vgg1d", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_vgg")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_vgg\checkpoints\best.pt", "--data-root", $BAL, "--output-dir", "..\results\fixed_full_ota_vgg")
    }
    "fixed_quick_ota_resnet_5ch" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $BAL, "--model-type", "resnet1d_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "cross_entropy", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_resnet_5ch")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_resnet_5ch\checkpoints\best.pt", "--data-root", $BAL, "--output-dir", "..\results\fixed_full_ota_resnet_5ch")
    }
    "fixed_quick_ota_multitask_5ch" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $BAL, "--model-type", "multitask_resnet1d_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "multitask", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_multitask_5ch")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_multitask_5ch\checkpoints\best.pt", "--data-root", $BAL, "--output-dir", "..\results\fixed_full_ota_multitask_5ch")
    }
    "fixed_quick_ota_multitask_margin_5ch" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $BAL, "--model-type", "multitask_resnet1d_margin_5ch", "--feature-mode", "iq_mag_ifreq_dphase", "--input-channels", "5", "--loss-type", "multitask_margin", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_multitask_margin_5ch")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_multitask_margin_5ch\checkpoints\best.pt", "--data-root", $BAL, "--output-dir", "..\results\fixed_full_ota_multitask_margin_5ch")
    }
    "fixed_quick_ota_rf_preprocessed_resnet" {
      Invoke-PythonModule @("-m", "src.train.train_oshea2018_extension", "--config", $CFG, "--data-root", $RF_BAL, "--model-type", "rf_preprocessed_resnet1d", "--feature-mode", "iq", "--input-channels", "5", "--loss-type", "cross_entropy", "--preset", "train", "--output-dir", "..\results\fixed_full_ota_rf_preprocessed_resnet")
      Invoke-PythonModule @("-m", "src.app.evaluate_oshea2018_extension", "--config", $CFG, "--checkpoint", "..\results\fixed_full_ota_rf_preprocessed_resnet\checkpoints\best.pt", "--data-root", $RF_BAL, "--output-dir", "..\results\fixed_full_ota_rf_preprocessed_resnet")
    }
    default {
      Write-Host "No Stage2 command mapped for $($row.model)"
    }
  }
  Write-Host "Completed Stage2 model: $($row.model)"
}

Write-Host ""
Write-Host "=== Summarize fixed full comparison ==="
Invoke-PythonModule @("-m", "src.analysis.summarize_oshea2018_extension", "--results-root", ".", "--profile", "fixed", "--output-dir", "..\results\fixed_full_comparison")

Write-Host ""
Write-Host ("Stage2 complete. Check fixed_full_* result folders and fixed_full_comparison under " + (Join-Path $SCRIPT_ROOT "results"))
} finally {
  Pop-Location
}

