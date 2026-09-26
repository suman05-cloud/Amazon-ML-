#!/bin/bash
set -e

# Make sure we are in the right directory
cd "$(dirname "$0")"

echo "=== Step 1: Preparing Massive Dataset ==="
# 9999999 guarantees we use the FULL dataset. Top-K 800 and Block-Cap 4000 maximize the candidate ceiling.
python -m amazon_ml prepare --sample-size 9999999 --top-k 800 --block-cap 4000 --work artifacts/experiment_large

echo "=== Step 2: Training Large Tree Model ==="
python -m amazon_ml train --work artifacts/experiment_large --threads 16

echo "=== Upgrading to 105-Feature Enhanced Tree ==="
# Extreme boosting rounds
python -m amazon_ml.improve --source artifacts/experiment_large --output artifacts/experiment_large_enhanced --threads 16 --rounds 4000

echo "=== Step 3: Preparing Neural Data ==="
python step3_fixed.py

echo "=== Step 4: Training Neural Model (CUDA) ==="
# Max sequence length (512), 10 epochs for deep convergence
python -u -m amazon_ml.neural --data artifacts/neural_data_large --output artifacts/neural_large --epochs 10 --length 512 --batch-size 8 --accumulation 8

echo "=== Building Raw Text for Rescore ==="
python -m amazon_ml.rescore --build-raw

echo "=== Rescoring Test Set with 16 Workers + GPU ==="
python -m amazon_ml.rescore --neural-model artifacts/neural_large --workers 16 --output output_final --gpu-batch 512

echo "=== PIPELINE COMPLETE ==="
