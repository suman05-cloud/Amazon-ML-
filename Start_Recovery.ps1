$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot '.venv_gpu/Scripts/python.exe') (Join-Path $PSScriptRoot 'scripts/job_supervisor.py')
Write-Host 'Local recovery supervisor launched. Status: artifacts/supervisor_status.json'
