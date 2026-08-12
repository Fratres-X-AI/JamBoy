# JamBoy laptop gate — never stop the campaign for WDAC-blocked rasterio DLLs.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }
$env:PYTHONPATH = "src"

Write-Host "==> rasterio probe"
$rasterioOk = $false
& $py -c "import rasterio; print(rasterio.__version__)" 2>$null
if ($LASTEXITCODE -eq 0) { $rasterioOk = $true; Write-Host "rasterio OK" }
else { Write-Host "HOST-BLOCKED: rasterio DLL (App Control). Full sim → Linux/pod. Continuing CPU unit subset." }

Write-Host "==> pytest (laptop subset)"
if ($rasterioOk) {
  & $py -m pytest -q
} else {
  # Units that do not import map_loader/geo_match
  & $py -m pytest -q `
    tests/test_ekf.py `
    tests/test_jamboy_ekf.py `
    tests/test_optical_flow.py `
    tests/test_realism.py `
    tests/test_stress_slow.py `
    tests/test_terminal_tracker.py
}
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

if (-not $rasterioOk) {
  Write-Host "OK - laptop subset PASS (full validate_sim HOST-BLOCKED → pod/Linux)"
  exit 0
}
Write-Host "==> generate + validate_sim"
& $py scripts/generate_dummy_data.py
& $py scripts/run_simulation.py --cpu --profile
& $py scripts/validate_sim.py
if ($LASTEXITCODE -ne 0) { throw "validate_sim failed" }
Write-Host "OK - JamBoy full laptop/sim gate PASS"
