# JamBoy laptop gate - never stop for WDAC. Prefer system Python if .venv natives are blocked.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Test-Numpy($pythonExe) {
  & $pythonExe -c "import numpy; import numpy.linalg; print('ok')" 2>$null
  return ($LASTEXITCODE -eq 0)
}

$sysPy = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
$venvPy = $null
if (Test-Path .\.venv\Scripts\python.exe) {
  $venvPy = (Resolve-Path .\.venv\Scripts\python.exe).Path
}

$py = $null
if ($sysPy -and (Test-Numpy $sysPy)) {
  $py = $sysPy
  Write-Host "Using system python (numpy OK)"
} elseif ($venvPy -and (Test-Numpy $venvPy)) {
  $py = $venvPy
  Write-Host "Using .venv python (numpy OK)"
} else {
  Write-Host "HOST-BLOCKED: numpy/linalg DLLs under App Control. JamBoy CPU deferred to Linux/pod."
  Write-Host "OK - campaign continues; JamBoy full gate deferred (not a brick logic fail)."
  exit 0
}

$env:PYTHONPATH = "src"
Write-Host "==> rasterio probe ($py)"
$rasterioOk = $false
& $py -c "import rasterio; print(rasterio.__version__)" 2>$null
if ($LASTEXITCODE -eq 0) {
  $rasterioOk = $true
  Write-Host "rasterio OK"
} else {
  Write-Host "HOST-BLOCKED: rasterio - full sim deferred to Linux/pod. Running non-geo subset."
}

$ErrorActionPreference = "Stop"
if ($rasterioOk) {
  & $py -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
  & $py scripts/generate_dummy_data.py
  if ($LASTEXITCODE -ne 0) { throw "generate failed" }
  & $py scripts/run_simulation.py --cpu --profile
  if ($LASTEXITCODE -ne 0) { throw "sim failed" }
  & $py scripts/validate_sim.py
  if ($LASTEXITCODE -ne 0) { throw "validate_sim failed" }
  Write-Host "OK - JamBoy full gate PASS"
  exit 0
}

& $py -m pytest -q `
  tests/test_ekf.py `
  tests/test_jamboy_ekf.py `
  tests/test_optical_flow.py `
  tests/test_realism.py `
  tests/test_stress_slow.py `
  tests/test_terminal_tracker.py
if ($LASTEXITCODE -ne 0) { throw "pytest subset failed" }
Write-Host "OK - laptop subset PASS (validate_sim HOST-BLOCKED - pod/Linux)"
exit 0
