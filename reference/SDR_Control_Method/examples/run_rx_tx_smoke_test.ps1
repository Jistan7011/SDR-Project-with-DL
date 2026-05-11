$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EXAMPLES = Join-Path $ROOT "examples"
$OUT = Join-Path $ROOT "output\smoke_test"
$RADIOCONDA = "C:\Users\qus70\radioconda"
$ACTIVATE = Join-Path $RADIOCONDA "Scripts\activate.bat"
$VENV_PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"

$CENTER = 433920000
$SAMPLE_RATE = 2400000
$SYMBOL_RATE = 5000
$OFFSET = 500000
$RX_GAIN = 30
$TX_VGA = 30
$TX_AMP = 0

Remove-Item $OUT -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

function Invoke-RadioConda {
  param([Parameter(Mandatory=$true)][string]$Command)
  cmd /c "call `"$ACTIVATE`" `"$RADIOCONDA`" && $Command"
  if ($LASTEXITCODE -ne 0) {
    throw "RadioConda command failed with exit code $LASTEXITCODE`: $Command"
  }
}

Write-Host ""
Write-Host "=== 1. HackRF info ==="
Invoke-RadioConda "hackrf_info"

Write-Host ""
Write-Host "=== 2. Soapy device diagnose ==="
Invoke-RadioConda "python `"$EXAMPLES\diagnose_soapy_devices.py`""

Write-Host ""
Write-Host "=== 3. RTL-SDR noise-only capture ==="
Invoke-RadioConda "python `"$EXAMPLES\rx_capture_soapy.py`" --output `"$OUT\noise_only.bin`" --seconds 2 --center-freq $CENTER --sample-rate $SAMPLE_RATE --rx-gain $RX_GAIN"

Write-Host ""
Write-Host "=== 4. HackRF standalone BPSK TX ==="
Invoke-RadioConda "python `"$EXAMPLES\tx_hackrf_transfer.py`" --modulation BPSK --seconds 1 --center-freq $CENTER --sample-rate $SAMPLE_RATE --symbol-rate $SYMBOL_RATE --baseband-offset-hz $OFFSET --tx-vga-gain $TX_VGA --tx-amp-gain $TX_AMP --seed 42"

foreach ($mod in @("BASK", "BFSK", "BPSK")) {
  $modOut = Join-Path $OUT $mod.ToLowerInvariant()
  Write-Host ""
  Write-Host "=== 5. RX/TX simultaneous capture: $mod ==="
  Invoke-RadioConda "python `"$EXAMPLES\rx_tx_capture_once.py`" --output-dir `"$modOut`" --modulation $mod --seconds 5 --tx-seconds 4.5 --noise-seconds 2 --rx-lead-seconds 0.5 --center-freq $CENTER --sample-rate $SAMPLE_RATE --symbol-rate $SYMBOL_RATE --baseband-offset-hz $OFFSET --rx-gain $RX_GAIN --tx-vga-gain $TX_VGA --tx-amp-gain $TX_AMP --seed 42"

  Write-Host ""
  Write-Host "=== 6. Analyze capture: $mod ==="
  & $VENV_PY "$EXAMPLES\analyze_iq_capture.py" `
    --noise "$modOut\noise_only.bin" `
    --capture "$modOut\$($mod.ToLowerInvariant())_capture.bin" `
    --sample-rate $SAMPLE_RATE `
    --active-start-seconds 1.1 `
    --active-duration-seconds 3.8 `
    --output "$modOut\analysis.json"
  if ($LASTEXITCODE -ne 0) {
    throw "Analysis failed for $mod"
  }
}

$leftovers = @(Get-Process | Where-Object {
  $_.ProcessName -match "python|hackrf|cmd"
})
if ($leftovers.Count -gt 0) {
  Write-Host ""
  Write-Host "Warning: possible leftover processes:"
  $leftovers | Select-Object Id,ProcessName,Path,StartTime | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Smoke test complete:"
Write-Host $OUT
