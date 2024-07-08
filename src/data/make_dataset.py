# src/data/make_dataset.py

import tensorflow as tf
import os
from pathlib import Path
import joblib

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def save_raw_data(raw_data_path):
    (x_train, y_train), (x_test, y_test) = load_data()
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump((x_train, y_train), raw_data_path / 'train_data.pkl')
    joblib.dump((x_test, y_test), raw_data_path / 'test_data.pkl')

if __name__ == "__main__":
    raw_data_path = os.path.join('data', 'raw', 'cifar-10-batches-py')
    save_raw_data(raw_data_path)
