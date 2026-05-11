$ErrorActionPreference = "Continue"

$ROOT = "D:\ai_projects\SDR\experiments\oshea2018"
$SRC = Join-Path $ROOT "sourcecode"
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.yaml"
$DATA = "..\data\ota_processed_clean"
$RF_DATA = "..\data\ota_rf_preprocessed_clean"
$LOG_DIR = Join-Path $ROOT "results\run_logs"
$LOG = Join-Path $LOG_DIR ("run_all_dl_120_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
trap {
    $message = "FATAL: $($_.Exception.Message)`n$($_.ScriptStackTrace)"
    Write-Host $message
    $message | Add-Content -Path $LOG -Encoding UTF8
    exit 1
}
Set-Location $SRC
$env:PYTHONPATH = $SRC

function Run-Step {
    param(
        [string] $Name,
        [scriptblock] $Command
    )
    $start = Get-Date
    Write-Host "`n===== START $Name : $start ====="
    "===== START $Name : $start =====" | Add-Content -Path $LOG -Encoding UTF8
    $global:LASTEXITCODE = 0
    & $Command 2>&1 | Tee-Object -FilePath $LOG -Append
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { $global:LASTEXITCODE }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    $end = Get-Date
    Write-Host "===== END $Name : $end elapsed=$([math]::Round(($end - $start).TotalMinutes, 2)) min ====="
    "===== END $Name : $end elapsed=$([math]::Round(($end - $start).TotalMinutes, 2)) min =====" | Add-Content -Path $LOG -Encoding UTF8
}

Run-Step "Oshea2018 raw-IQ ResNet train" {
    & $PY -m src.train.train_oshea2018 `
        --config $CFG `
        --data-root $DATA `
        --model-type oshea2018_resnet1d `
        --preset train `
        --output-dir ..\results\ota_resnet
}

Run-Step "Oshea2018 raw-IQ ResNet eval" {
    & $PY -m src.app.evaluate_oshea2018 `
        --config $CFG `
        --checkpoint ..\results\ota_resnet\checkpoints\best.pt `
        --data-root $DATA `
        --output-dir ..\results\ota_resnet
}

Run-Step "Oshea2018 raw-IQ VGG train" {
    & $PY -m src.train.train_oshea2018 `
        --config $CFG `
        --data-root $DATA `
        --model-type oshea2018_vgg1d `
        --preset train `
        --output-dir ..\results\ota_vgg
}

Run-Step "Oshea2018 raw-IQ VGG eval" {
    & $PY -m src.app.evaluate_oshea2018 `
        --config $CFG `
        --checkpoint ..\results\ota_vgg\checkpoints\best.pt `
        --data-root $DATA `
        --output-dir ..\results\ota_vgg
}

Run-Step "Exp5-style 5ch ResNet train" {
    & $PY -m src.train.train_oshea2018_extension `
        --config $CFG `
        --data-root $DATA `
        --model-type resnet1d_5ch `
        --feature-mode iq_mag_ifreq_dphase `
        --input-channels 5 `
        --loss-type cross_entropy `
        --preset train `
        --output-dir ..\results\ota_resnet_5ch
}

Run-Step "Exp5-style 5ch ResNet eval" {
    & $PY -m src.app.evaluate_oshea2018_extension `
        --config $CFG `
        --checkpoint ..\results\ota_resnet_5ch\checkpoints\best.pt `
        --data-root $DATA `
        --output-dir ..\results\ota_resnet_5ch
}

Run-Step "Exp8-style 5ch MultiTaskResNet train" {
    & $PY -m src.train.train_oshea2018_extension `
        --config $CFG `
        --data-root $DATA `
        --model-type multitask_resnet1d_5ch `
        --feature-mode iq_mag_ifreq_dphase `
        --input-channels 5 `
        --loss-type multitask `
        --preset train `
        --output-dir ..\results\ota_multitask_5ch
}

Run-Step "Exp8-style 5ch MultiTaskResNet eval" {
    & $PY -m src.app.evaluate_oshea2018_extension `
        --config $CFG `
        --checkpoint ..\results\ota_multitask_5ch\checkpoints\best.pt `
        --data-root $DATA `
        --output-dir ..\results\ota_multitask_5ch
}

Run-Step "Exp8-style 5ch MultiTaskResNet margin train" {
    & $PY -m src.train.train_oshea2018_extension `
        --config $CFG `
        --data-root $DATA `
        --model-type multitask_resnet1d_margin_5ch `
        --feature-mode iq_mag_ifreq_dphase `
        --input-channels 5 `
        --loss-type multitask_margin `
        --preset train `
        --output-dir ..\results\ota_multitask_margin_5ch
}

Run-Step "Exp8-style 5ch MultiTaskResNet margin eval" {
    & $PY -m src.app.evaluate_oshea2018_extension `
        --config $CFG `
        --checkpoint ..\results\ota_multitask_margin_5ch\checkpoints\best.pt `
        --data-root $DATA `
        --output-dir ..\results\ota_multitask_margin_5ch
}

Run-Step "Exp9-style RF preprocessing dataset" {
    & $PY -m src.dataset.prepare_oshea2018_rf_preprocessed `
        --config $CFG `
        --source-root $DATA `
        --output-root $RF_DATA
}

Run-Step "Exp9-style RF-preprocessed ResNet train" {
    & $PY -m src.train.train_oshea2018_extension `
        --config $CFG `
        --data-root $RF_DATA `
        --model-type rf_preprocessed_resnet1d `
        --feature-mode iq `
        --input-channels 5 `
        --loss-type cross_entropy `
        --preset train `
        --output-dir ..\results\ota_rf_preprocessed_resnet
}

Run-Step "Exp9-style RF-preprocessed ResNet eval" {
    & $PY -m src.app.evaluate_oshea2018_extension `
        --config $CFG `
        --checkpoint ..\results\ota_rf_preprocessed_resnet\checkpoints\best.pt `
        --data-root $RF_DATA `
        --output-dir ..\results\ota_rf_preprocessed_resnet
}

Run-Step "Summary" {
    & $PY -m src.analysis.summarize_oshea2018_extension `
        --results-root . `
        --output-dir ..\results\oshea2018_extension_summary
}

Write-Host "`nAll deep-learning runs complete. Log: $LOG"
