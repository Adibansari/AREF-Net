# src/data/preprocess_data.py

import os
import joblib
from pathlib import Path

def preprocess_data(raw_data_path, processed_data_path):
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    (x_train, y_train) = joblib.load(raw_data_path / 'train_data.pkl')
    (x_test, y_test) = joblib.load(raw_data_path / 'test_data.pkl')

    # Normalize the dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Save processed data
    joblib.dump((x_train, y_train), processed_data_path / 'train_data.pkl')
    joblib.dump((x_test, y_test), processed_data_path / 'test_data.pkl')

if __name__ == "__main__":
    raw_data_path = os.path.join('data', 'raw', 'cifar-10-batches-py')
    processed_data_path = os.path.join('data', 'processed')
    preprocess_data(raw_data_path, processed_data_path)
