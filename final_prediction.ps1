Write-Host "=== Building Raw Text for Rescore ==="
.\.venv\Scripts\python.exe -m amazon_ml.rescore --build-raw

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Rescoring Test Set with 16 Workers + GPU ==="
    .\.venv_gpu\Scripts\python.exe -m amazon_ml.rescore --neural-model artifacts/neural_large --workers 16 --output output_final --gpu-batch 512
}
