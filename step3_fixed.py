import os
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from amazon_ml.neural_data import prepare as prepare_neural_data

prepare_neural_data(
    data="data set/student_resource/dataset",
    output="artifacts/neural_data_large",
    source="artifacts/experiment_large_enhanced",
    teacher="artifacts/experiment_large_enhanced",
    confirmation="artifacts/confirmation_v3"
)
