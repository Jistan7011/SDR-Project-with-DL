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
$PREFLIGHT = "..\results\fixed_preflight_analysis\fixed_preflight_quality.json"

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

if (!(Test-Path $PREFLIGHT)) {
  throw "Fixed preflight result not found. Run run_fixed_preflight.ps1 first and inspect fixed_preflight_summary.md."
}

$preflightObj = Get-Content $PREFLIGHT -Raw | ConvertFrom-Json
$conditions = @(
  $preflightObj.conditions |
    Where-Object { $_.recommend_for_full_collection } |
    Sort-Object -Property recommendation_score -Descending |
    ForEach-Object {
      @{
        tx = [double]$_.tx_vga_gain
        rx = [double]$_.rx_gain
        offset = [double]$_.baseband_offset_hz
        score = [double]$_.recommendation_score
      }
    }
)

if ($conditions.Count -eq 0) {
  throw "Stage0 blocked: no fixed preflight condition reached BASK/BPSK pass-rate gate. Do not collect full data yet."
}

Write-Host "Selected fixed collection conditions from preflight:"
foreach ($c in $conditions) {
  Write-Host ("  tx_vga={0}, rx_gain={1}, offset={2}, score={3:N4}" -f $c.tx, $c.rx, $c.offset, $c.score)
}

Remove-Item $RAW -Recurse -Force -ErrorAction SilentlyContinue

for ($i = 1; $i -le 20; $i++) {
  $sid = "session_{0:D3}" -f $i
  $c = $conditions[($i - 1) % $conditions.Count]
  Write-Host ""
  Write-Host "=== Fixed collection ${sid}: tx_vga=$($c.tx), rx_gain=$($c.rx), offset=$($c.offset) ==="
  Invoke-PythonModule @(
    "-m", "src.experiment.run_oshea2018_capture_session",
    "--config", $CFG,
    "--session-id", $sid,
    "--output-root", $RAW,
    "--captures-per-class", "10",
    "--tx-vga-gain", [string]$c.tx,
    "--tx-amp-gain", "0",
    "--rx-gain", [string]$c.rx,
    "--baseband-offset-hz", [string]$c.offset,
    "--max-retries", "3"
  )
}

Write-Host ""
Write-Host "Fixed raw collection complete:"
Write-Host (Join-Path $SCRIPT_ROOT "data\raw_ota_fixed")
} finally {
  Pop-Location
}

