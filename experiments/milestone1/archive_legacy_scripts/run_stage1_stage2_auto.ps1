$ErrorActionPreference = "Continue"

$ROOT = "D:\ai_projects\SDR\experiments\oshea2018"
$SRC = Join-Path $ROOT "sourcecode"
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.yaml"
$DATA = "..\data\ota_processed_clean"
$QUICK = "..\data\ota_processed_quick"
$RF_QUICK = "..\data\ota_rf_preprocessed_quick"
$RF_DATA = "..\data\ota_rf_preprocessed_clean"
$LOG_DIR = Join-Path $ROOT "results\run_logs"
$LOG = Join-Path $LOG_DIR ("run_stage1_stage2_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
Set-Location $SRC
$env:PYTHONPATH = $SRC

function Log-Line {
    param([string] $Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    $line | Add-Content -Path $LOG -Encoding UTF8
}

function Run-Step {
    param(
        [string] $Name,
        [scriptblock] $Command
    )
    $start = Get-Date
    Log-Line "START $Name"
    $global:LASTEXITCODE = 0
    & $Command 2>&1 | Tee-Object -FilePath $LOG -Append
    $exitCode = if ($null -eq $global:LASTEXITCODE) { 0 } else { $global:LASTEXITCODE }
    if ($exitCode -ne 0) {
        Log-Line "FAILED $Name exit=$exitCode"
        throw "$Name failed with exit code $exitCode"
    }
    $elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 2)
    Log-Line "DONE $Name elapsed=${elapsed}min"
}

function Reset-Dir {
    param([string] $Path)
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Copy-BalancedSubset {
    param(
        [string] $SourceRoot,
        [string] $DestRoot,
        [int] $TrainPerClass = 5000,
        [int] $ValPerClass = 3000,
        [int] $TestPerClass = 5000
    )
    Reset-Dir $DestRoot
    foreach ($split in @("train", "val", "test")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $DestRoot $split) | Out-Null
    }
    foreach ($split in @("train", "val", "test")) {
        $limit = if ($split -eq "train") { $TrainPerClass } elseif ($split -eq "val") { $ValPerClass } else { $TestPerClass }
        foreach ($mod in @("bask", "bfsk", "bpsk")) {
            $files = @(Get-ChildItem (Join-Path $SourceRoot $split) -Filter "*_${mod}_*.npz" | Sort-Object Name)
            if ($files.Count -eq 0) {
                throw "No files for split=$split mod=$mod"
            }
            $take = [Math]::Min($limit, $files.Count)
            if ($take -eq $files.Count) {
                $selected = $files
            } else {
                $selected = New-Object System.Collections.Generic.List[object]
                for ($i = 0; $i -lt $take; $i++) {
                    $idx = [int][Math]::Round($i * (($files.Count - 1) / [double]($take - 1)))
                    $selected.Add($files[$idx])
                }
            }
            foreach ($file in $selected) {
                $dest = Join-Path (Join-Path $DestRoot $split) $file.Name
                try {
                    New-Item -ItemType HardLink -Path $dest -Target $file.FullName -Force | Out-Null
                } catch {
                    Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
                }
            }
            Log-Line "Quick subset split=$split mod=$mod copied=$take source=$($files.Count)"
        }
    }
}

function Eval-Score {
    param([string] $Path)
    if (!(Test-Path $Path)) { return "missing" }
    $j = Get-Content $Path -Raw | ConvertFrom-Json
    $acc = [double]$j.accuracy
    $f1 = if ($null -ne $j.macro_f1) { [double]$j.macro_f1 } else { [double]$j.'macro avg'.'f1-score' }
    $abs = if ($null -ne $j.bfsk_bpsk_to_bask_rate) { [double]$j.bfsk_bpsk_to_bask_rate } else { 0.0 }
    return ("accuracy={0:N4} macro_f1={1:N4} absorption={2:N4}" -f $acc, $f1, $abs)
}

Log-Line "Stage1+Stage2 auto run begins"
Log-Line "Config: quick smoke uses subset and --preset smoke; stage2 uses full data and config epochs=8"

Run-Step "Stage1 quick subset creation" {
    Copy-BalancedSubset -SourceRoot $DATA -DestRoot $QUICK -TrainPerClass 5000 -ValPerClass 3000 -TestPerClass 5000
}

Run-Step "Stage1 quick RF preprocessing subset" {
    & $PY -m src.dataset.prepare_oshea2018_rf_preprocessed --config $CFG --source-root $QUICK --output-root $RF_QUICK
}

$stage1 = @(
    @{Name="quick_raw_resnet"; Kind="raw"; Model="oshea2018_resnet1d"; Train="src.train.train_oshea2018"; Eval="src.app.evaluate_oshea2018"; Data=$QUICK; Out="..\results\quick_ota_resnet"},
    @{Name="quick_raw_vgg"; Kind="raw"; Model="oshea2018_vgg1d"; Train="src.train.train_oshea2018"; Eval="src.app.evaluate_oshea2018"; Data=$QUICK; Out="..\results\quick_ota_vgg"},
    @{Name="quick_5ch_resnet"; Kind="ext"; Model="resnet1d_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="cross_entropy"; Data=$QUICK; Out="..\results\quick_ota_resnet_5ch"},
    @{Name="quick_multitask"; Kind="ext"; Model="multitask_resnet1d_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="multitask"; Data=$QUICK; Out="..\results\quick_ota_multitask_5ch"},
    @{Name="quick_multitask_margin"; Kind="ext"; Model="multitask_resnet1d_margin_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="multitask_margin"; Data=$QUICK; Out="..\results\quick_ota_multitask_margin_5ch"},
    @{Name="quick_rf_preprocessed_resnet"; Kind="ext"; Model="rf_preprocessed_resnet1d"; Feature="iq"; Loss="cross_entropy"; Data=$RF_QUICK; Out="..\results\quick_ota_rf_preprocessed_resnet"}
)

foreach ($run in $stage1) {
    Reset-Dir $run.Out
    if ($run.Kind -eq "raw") {
        Run-Step "Stage1 train $($run.Name)" {
            & $PY -m $run.Train --config $CFG --data-root $run.Data --model-type $run.Model --preset smoke --output-dir $run.Out
        }
        Run-Step "Stage1 eval $($run.Name)" {
            & $PY -m $run.Eval --config $CFG --checkpoint (Join-Path $run.Out "checkpoints\best.pt") --data-root $run.Data --output-dir $run.Out
        }
    } else {
        Run-Step "Stage1 train $($run.Name)" {
            & $PY -m src.train.train_oshea2018_extension --config $CFG --data-root $run.Data --model-type $run.Model --feature-mode $run.Feature --input-channels 5 --loss-type $run.Loss --preset smoke --output-dir $run.Out
        }
        Run-Step "Stage1 eval $($run.Name)" {
            & $PY -m src.app.evaluate_oshea2018_extension --config $CFG --checkpoint (Join-Path $run.Out "checkpoints\best.pt") --data-root $run.Data --output-dir $run.Out
        }
    }
    Log-Line "Stage1 result $($run.Name): $(Eval-Score (Join-Path $run.Out 'logs\eval_test.json'))"
}

$stage2 = @(
    @{Name="full_raw_resnet"; Kind="raw"; Model="oshea2018_resnet1d"; Train="src.train.train_oshea2018"; Eval="src.app.evaluate_oshea2018"; Data=$DATA; Out="..\results\ota_resnet"},
    @{Name="full_raw_vgg"; Kind="raw"; Model="oshea2018_vgg1d"; Train="src.train.train_oshea2018"; Eval="src.app.evaluate_oshea2018"; Data=$DATA; Out="..\results\ota_vgg"},
    @{Name="full_5ch_resnet"; Kind="ext"; Model="resnet1d_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="cross_entropy"; Data=$DATA; Out="..\results\ota_resnet_5ch"},
    @{Name="full_multitask"; Kind="ext"; Model="multitask_resnet1d_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="multitask"; Data=$DATA; Out="..\results\ota_multitask_5ch"},
    @{Name="full_multitask_margin"; Kind="ext"; Model="multitask_resnet1d_margin_5ch"; Feature="iq_mag_ifreq_dphase"; Loss="multitask_margin"; Data=$DATA; Out="..\results\ota_multitask_margin_5ch"}
)

Run-Step "Stage2 full RF preprocessing dataset" {
    & $PY -m src.dataset.prepare_oshea2018_rf_preprocessed --config $CFG --source-root $DATA --output-root $RF_DATA
}
$stage2 += @{Name="full_rf_preprocessed_resnet"; Kind="ext"; Model="rf_preprocessed_resnet1d"; Feature="iq"; Loss="cross_entropy"; Data=$RF_DATA; Out="..\results\ota_rf_preprocessed_resnet"}

foreach ($run in $stage2) {
    Reset-Dir $run.Out
    if ($run.Kind -eq "raw") {
        Run-Step "Stage2 train $($run.Name)" {
            & $PY -m $run.Train --config $CFG --data-root $run.Data --model-type $run.Model --preset train --output-dir $run.Out
        }
        Run-Step "Stage2 eval $($run.Name)" {
            & $PY -m $run.Eval --config $CFG --checkpoint (Join-Path $run.Out "checkpoints\best.pt") --data-root $run.Data --output-dir $run.Out
        }
    } else {
        Run-Step "Stage2 train $($run.Name)" {
            & $PY -m src.train.train_oshea2018_extension --config $CFG --data-root $run.Data --model-type $run.Model --feature-mode $run.Feature --input-channels 5 --loss-type $run.Loss --preset train --output-dir $run.Out
        }
        Run-Step "Stage2 eval $($run.Name)" {
            & $PY -m src.app.evaluate_oshea2018_extension --config $CFG --checkpoint (Join-Path $run.Out "checkpoints\best.pt") --data-root $run.Data --output-dir $run.Out
        }
    }
    Log-Line "Stage2 result $($run.Name): $(Eval-Score (Join-Path $run.Out 'logs\eval_test.json'))"
}

Run-Step "Stage2 summary" {
    & $PY -m src.analysis.summarize_oshea2018_extension --results-root . --output-dir ..\results\oshea2018_extension_summary
}

Log-Line "Stage1+Stage2 auto run complete"
Log-Line "Summary path: D:\ai_projects\SDR\experiments\oshea2018\results\oshea2018_extension_summary\oshea2018_extension_summary.md"
