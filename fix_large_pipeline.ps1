Write-Host "=== Upgrading to 105-Feature Enhanced Tree ==="
.\.venv\Scripts\python.exe -m amazon_ml.improve --source artifacts/experiment_large --output artifacts/experiment_large_enhanced --threads 16 --rounds 2500

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Step 3: Preparing Neural Data ==="
    .\.venv\Scripts\python.exe step3_fixed.py

    if ($LASTEXITCODE -eq 0) {
        Write-Host "=== Step 4: Training Neural Model (CUDA) ==="
        .\.venv_gpu\Scripts\python.exe -u -m amazon_ml.neural --data artifacts/neural_data_large --output artifacts/neural_large --epochs 8 --length 384 --batch-size 8 --accumulation 4
    }
}
