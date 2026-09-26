import os
import sys
from pathlib import Path

# 1. Prepare Large Tree Data
print("=== Step 1: Preparing Massive Dataset (500,000 samples, top-k 400) ===")
os.system(r'.\.venv\Scripts\python.exe -m amazon_ml prepare --sample-size 500000 --top-k 400 --block-cap 2000 --work artifacts/experiment_large')

# 2. Train Large Tree Model
print("=== Step 2: Training Large Tree Model ===")
os.system(r'.\.venv\Scripts\python.exe -m amazon_ml train --work artifacts/experiment_large --threads 16')

# 3. Generate Neural Data using the new massive tree model
print("=== Step 3: Preparing Neural Data ===")
import sys
sys.path.append(str(Path("src").resolve()))
from amazon_ml.neural_data import prepare as prepare_neural_data

# The source is where cases/entities are (the new large tree experiment)
# The teacher is where features.npy and probabilities.npy are (the new large tree experiment)
# We can use the baseline confirmation for now as the evaluation set
prepare_neural_data(
    data="data set/student_resource/dataset",
    output="artifacts/neural_data_large",
    source="artifacts/experiment_large",
    teacher="artifacts/experiment_large",
    confirmation="artifacts/confirmation_v3" 
)

# 4. Train the Neural Model (CUDA)
print("=== Step 4: Training Neural Model (CUDA) ===")
os.system(r'.\.venv_gpu\Scripts\python.exe -u -m amazon_ml.neural --data artifacts/neural_data_large --output artifacts/neural_large')

print("=== ALL TRAINING COMPLETE ===")
