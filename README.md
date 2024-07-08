# AREF-Net: Bridging Residual, Efficient, and Attention-Based Architectures for Image Classification

This repository contains the implementation of AREF-Net, a model designed for image classification using the CIFAR-10 dataset. AREF-Net integrates Residual Networks, EfficientNet, and Attention Mechanisms to achieve high performance.

## Introduction

AREF-Net is a hybrid model that combines the strengths of Residual Networks, EfficientNet, and Attention Mechanisms to provide a robust solution for image classification tasks. This project includes the training pipeline, model deployment using FastAPI, and instructions to reproduce the results.

## Features

- Hybrid architecture combining Residual Networks, EfficientNet, and Attention Mechanisms.
- Training pipeline using DVC.
- Model deployment with FastAPI.
## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- DVC
- Docker (optional, for containerization)
- FastAPI

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/Adibansari/AREF-Net.git
    cd AREF-Net
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up DVC:

    ```bash
    dvc pull
    ```

## Usage

### Training the Model

To initiate the training pipeline:

```bash
dvc repro
```
Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Deploying the Model
To deploy the model using FastAPI, run:
```bash
uvicorn app.main:app --reload
```
## Making Predictions
You can make predictions by sending a POST request to the FastAPI endpoint with an image file. For example, using curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_image.jpg"

```
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.
## Citation
If you use this code in your research, please cite our paper:
```bash
A. Ansari, G. Marken, S. Shobhit, and P. Dongre, "AREF-Net: Bridging Residual, Efficient, and Attention-Based Architectures for Image Classification," in 2023 International Conference on Advanced Computing & Communication Technologies (ICACCTech), Banur, India, 2023, pp. 450-456. doi: 10.1109/ICACCTech61146.2023.00080.
```
