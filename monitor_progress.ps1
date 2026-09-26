param([int]$RefreshSeconds = 10)
$ErrorActionPreference = "Continue"
$projectRoot = $PSScriptRoot
while ($true) {
    Clear-Host
    Write-Host "Amazon ML progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Ctrl+C stops this monitor only. It does not stop prediction or training."
    $processed = 0L
    Get-ChildItem -LiteralPath (Join-Path $projectRoot 'output/logs') -Filter 'shard-*.log' -ErrorAction SilentlyContinue | ForEach-Object {
        $lastLine = Get-Content -LiteralPath $_.FullName -Tail 1
        Write-Host $lastLine
        if ($lastLine -match '([\d,]+) references processed') {
            $processed += [long]($Matches[1] -replace ',', '')
        }
    }
    Write-Host ("Baseline prediction: {0:N0} / 1,732,544 ({1:P1})" -f $processed, ($processed / 1732544.0))
    foreach ($entry in @(
        @('Baseline', 'output/run_status.json'),
        @('GPU neural', 'artifacts/neural_v1/status.json'),
        @('Larger GPU model', 'artifacts/large_v1/status.json'),
        @('Retrieval study', 'artifacts/retrieval_study_v1/status.json')
    )) {
        $statusPath = Join-Path $projectRoot $entry[1]
        if (Test-Path -LiteralPath $statusPath) {
            try {
                $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
                Write-Host "`n$($entry[0]):"
                $state | Select-Object state,epoch,epochs,pairs_seen,pairs_per_epoch,pairs_per_second,new_references,total_new_references,indexed_targets,depth,round | Format-List | Out-Host
            } catch {
                Write-Host 'Status is being updated; retrying on next refresh.'
            }
        }
    }
    Write-Host 'GPU utilization, memory, power:'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
    Start-Sleep -Seconds $RefreshSeconds
}
