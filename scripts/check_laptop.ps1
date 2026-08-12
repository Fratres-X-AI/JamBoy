# JamBoy laptop gate — never stop the campaign for WDAC-blocked rasterio DLLs.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path .\.venv\Scripts\python.exe) { (Resolve-Path .\.venv\Scripts\python.exe).Path } else { "python" }
$env:PYTHONPATH = "src"

Write-Host "==> rasterio probe"
$rasterioOk = $false
$probe = & $py -c "import rasterio; print(rasterio.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
  $rasterioOk = $true
  Write-Host "rasterio OK $probe"
} else {
  Write-Host "HOST-BLOCKED: rasterio DLL (App Control). Full sim → Linux/pod. Continuing CPU unit subset."
}

Write-Host "==> pytest (laptop subset)"
$ErrorActionPreference = "Stop"
if ($rasterioOk) {
  & $py -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
} else {
  & $py -m pytest -q `
    tests/test_ekf.py `
    tests/test_jamboy_ekf.py `
    tests/test_optical_flow.py `
    tests/test_realism.py `
    tests/test_stress_slow.py `
    tests/test_terminal_tracker.py
  if ($LASTEXITCODE -ne 0) { throw "pytest subset failed" }
  Write-Host "OK - laptop subset PASS (full validate_sim HOST-BLOCKED → pod/Linux)"
  exit 0
}

Write-Host "==> generate + validate_sim"
& $py scripts/generate_dummy_data.py
if ($LASTEXITCODE -ne 0) { throw "generate_dummy_data failed" }
& $py scripts/run_simulation.py --cpu --profile
if ($LASTEXITCODE -ne 0) { throw "run_simulation failed" }
& $py scripts/validate_sim.py
if ($LASTEXITCODE -ne 0) { throw "validate_sim failed" }
Write-Host "OK - JamBoy full laptop/sim gate PASS"
