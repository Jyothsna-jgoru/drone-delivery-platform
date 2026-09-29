$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& .\.venv\Scripts\python -m pip install -e ".[dev]"
Push-Location apps\web
npm install --cache .npm-cache
Pop-Location

$apiProcess = Start-Process -FilePath "$projectRoot\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru

Write-Host "API:       http://127.0.0.1:8000/docs"
Write-Host "Dashboard: http://127.0.0.1:5173"
Write-Host "Press Ctrl+C to stop the dashboard."

try {
    Set-Location "$projectRoot\apps\web"
    npm run dev -- --host 127.0.0.1
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id
    }
}
