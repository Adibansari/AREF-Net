# src/features/build_features.py
import joblib
import numpy as np
from keras.utils import to_categorical
from pathlib import Path

def preprocess_data(input_file, output_file):
    data = joblib.load(input_file)
    x, y = data
    x = x.astype('float32') / 255.0
    y = to_categorical(y, 10)

    joblib.dump((x, y), output_file)

if __name__ == "__main__":
    input_files = [Path("data/raw/cifar-10-batches-py/train.pkl"), Path("data/raw/cifar-10-batches-py/test.pkl")]
    output_files = [Path("data/processed/train.pkl"), Path("data/processed/test.pkl")]

    for input_file, output_file in zip(input_files, output_files):
        preprocess_data(input_file, output_file)
