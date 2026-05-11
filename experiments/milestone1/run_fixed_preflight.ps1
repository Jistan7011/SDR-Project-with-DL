$ErrorActionPreference = "Stop"

$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location (Join-Path $SCRIPT_ROOT "sourcecode")
try {
$env:PYTHONPATH = (Get-Location).Path
$REPO_ROOT = (Resolve-Path (Join-Path $SCRIPT_ROOT "..\..")).Path
$PY = Join-Path $REPO_ROOT ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }
$CFG = "..\config\config.milestone1.fixed.yaml"
$RAW = "..\data\raw_ota_fixed_preflight"
$OUT = "..\results\fixed_preflight_analysis"

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

Remove-Item $RAW -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $OUT -Recurse -Force -ErrorAction SilentlyContinue

$conditions = @()
$sessionIndex = 1
foreach ($offset in @(250000, 500000)) {
  foreach ($tx in @(25, 30, 35)) {
    foreach ($rx in @(25, 30, 35)) {
      $conditions += @{
        sid = "session_{0:D3}" -f $sessionIndex
        tx = $tx
        rx = $rx
        offset = $offset
      }
      $sessionIndex += 1
    }
  }
}

foreach ($c in $conditions) {
  Write-Host ""
  Write-Host "=== Fixed preflight $($c.sid): tx_vga=$($c.tx), rx_gain=$($c.rx), offset=$($c.offset) ==="
  Invoke-PythonModule @(
    "-m", "src.experiment.run_oshea2018_capture_session",
    "--config", $CFG,
    "--session-id", $c.sid,
    "--output-root", $RAW,
    "--captures-per-class", "2",
    "--tx-vga-gain", [string]$c.tx,
    "--tx-amp-gain", "0",
    "--rx-gain", [string]$c.rx,
    "--baseband-offset-hz", [string]$c.offset,
    "--max-retries", "3"
  )
}

Write-Host ""
Write-Host "=== Analyze fixed preflight ==="
Invoke-PythonModule @(
  "-m", "src.analysis.analyze_oshea2018_fixed_preflight",
  "--config", $CFG,
  "--raw-root", $RAW,
  "--output-dir", $OUT
)

Write-Host ""
Write-Host "Preflight summary:"
Write-Host (Join-Path $SCRIPT_ROOT "results\fixed_preflight_analysis\fixed_preflight_summary.md")
} finally {
  Pop-Location
}

