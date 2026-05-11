$ErrorActionPreference = "Stop"

$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location (Join-Path $SCRIPT_ROOT "sourcecode")
try {
$env:PYTHONPATH = (Get-Location).Path
$REPO_ROOT = (Resolve-Path (Join-Path $SCRIPT_ROOT "..\..")).Path
$PY = Join-Path $REPO_ROOT ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }
$RADIOCONDA = if ($env:RADIOCONDA_ROOT) { $env:RADIOCONDA_ROOT } else { "C:\Users\qus70\radioconda" }
$RADIOCONDA_ACTIVATE = Join-Path $RADIOCONDA "Scripts\activate.bat"
if (!(Test-Path $RADIOCONDA_ACTIVATE)) {
  throw "RADIOCONDA_ROOT is not set or activate.bat was not found. Set `$env:RADIOCONDA_ROOT to your radioconda path."
}
$CFG = "..\config\config.milestone1.fixed.yaml"
$OUT = "..\data\hardware_check"

Remove-Item $OUT -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

Write-Host ""
Write-Host "=== Stage0: SoapySDR device diagnose ==="
cmd /c "call `"$RADIOCONDA_ACTIVATE`" `"$RADIOCONDA`" && python -m src.sdr.diagnose"

Write-Host ""
Write-Host "=== Stage0: RTL-SDR noise-only capture ==="
cmd /c "call `"$RADIOCONDA_ACTIVATE`" `"$RADIOCONDA`" && python -m src.sdr.capture_iq --config $CFG --output ..\data\hardware_check\noise_only.bin --seconds 2.0 --sample-rate 2400000 --rx-gain 30"

foreach ($mod in @("BASK", "BFSK", "BPSK")) {
  Write-Host ""
  Write-Host "=== Stage0: HackRF standalone TX $mod ==="
  cmd /c "call `"$RADIOCONDA_ACTIVATE`" `"$RADIOCONDA`" && python -m src.sdr.hackrf_tx_oshea2018 --config $CFG --modulation $mod --seconds 1.0 --seed 42 --tx-vga-gain 30 --tx-amp-gain 0 --baseband-offset-hz 500000 --backend hackrf_transfer"
}

Write-Host ""
Write-Host "=== Stage0: RX/TX simultaneous one-capture session ==="
& $PY -m src.experiment.run_oshea2018_capture_session `
  --config $CFG `
  --session-id session_001 `
  --output-root $OUT `
  --captures-per-class 1 `
  --tx-vga-gain 30 `
  --tx-amp-gain 0 `
  --rx-gain 30 `
  --baseband-offset-hz 500000 `
  --max-retries 1 `
  --tx-backend hackrf_transfer `
  --radioconda-root $RADIOCONDA

if (!(Test-Path "..\data\hardware_check\session_001\metadata.json")) {
  throw "Stage0 failed: simultaneous capture did not produce metadata.json."
}

$leftovers = @(Get-Process | Where-Object {
  $_.Path -like (Join-Path $RADIOCONDA "python.exe") -or $_.ProcessName -eq "cmd"
})
if ($leftovers.Count -gt 0) {
  Write-Host "Warning: possible leftover SDR subprocesses:"
  $leftovers | Select-Object Id,ProcessName,CPU,StartTime,Path | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Stage0 complete. Hardware check data:"
Write-Host (Join-Path $SCRIPT_ROOT "data\hardware_check")
} finally {
  Pop-Location
}
