$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
& .\.venv\Scripts\python -m pip install -e ".[dev]"
Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "apps.api.main:app", "--reload"
Set-Location apps\web
npm install
npm run dev

