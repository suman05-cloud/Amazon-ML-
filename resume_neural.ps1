Copy-Item -Path "artifacts\experiment_large\cases.jsonl" -Destination "artifacts\experiment_large_enhanced\cases.jsonl"

Write-Host "=== Step 3: Preparing Neural Data ==="
.\.venv\Scripts\python.exe step3_fixed.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Step 4: Training Neural Model (CUDA) ==="
    .\.venv_gpu\Scripts\python.exe -u -m amazon_ml.neural --data artifacts/neural_data_large --output artifacts/neural_large
}
