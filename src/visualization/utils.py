# src/utils.py
import joblib
from pathlib import Path

def save_model(model, path):
    joblib.dump(model, Path(path))

def load_model(path):
    return joblib.load(Path(path))
