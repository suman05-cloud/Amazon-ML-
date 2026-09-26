.\.venv\Scripts\python.exe step3.py
if ($LASTEXITCODE -eq 0) {
    print "=== Step 4: Training Neural Model (CUDA) ==="
    .\.venv_gpu\Scripts\python.exe -u -m amazon_ml.neural --data artifacts/neural_data_large --output artifacts/neural_large
}
