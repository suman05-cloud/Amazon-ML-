param(
    [string]$Data = 'data set/student_resource/dataset',
    [int]$SampleSize = 8000,
    [int]$Threads = 4,
    [string]$Work = 'artifacts/experiment_v2',
    [string]$Index = 'artifacts/test.sqlite',
    [string]$Output = 'output'
)
$ErrorActionPreference = 'Stop'
$PythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
function Run-Step {
    param([string[]]$Arguments)
    & $PythonPath -m amazon_ml @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
}
Push-Location $PSScriptRoot
try {
    Run-Step -Arguments @('prepare', '--data', $Data, '--work', $Work, '--sample-size', "$SampleSize")
    Run-Step -Arguments @('train', '--work', $Work, '--threads', "$Threads")
    Run-Step -Arguments @('build-index', '--data', $Data, '--index', $Index)
    Run-Step -Arguments @('predict', '--data', $Data, '--index', $Index, '--model', "$Work/model", '--output', $Output, '--threads', "$Threads")
    Run-Step -Arguments @('validate', '--data', $Data, '--output', $Output)
} finally {
    Pop-Location
}
