# Turbofan Engine RUL Prediction (upskillcampus_turbofan)

Predict Remaining Useful Life (RUL) of turbofan (jet) engines using time-series sensor data. This project ingests historical multivariate sensor readings, processes them, trains predictive models, and provides a user-facing interface for inference.

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Data Preparation](#data-preparation)  
  - [Training & Evaluation](#training--evaluation)  
  - [Running Inference / Application](#running-inference--application)  
- [Model & Methods](#model--methods)  
- [Usage Examples](#usage-examples)  
- [Results & Metrics](#results--metrics)  
- [Future Work / Roadmap](#future-work--roadmap)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)

---

## Overview

Turbofan engines degrade over time, and predicting their Remaining Useful Life (RUL) is critical for condition-based maintenance and avoiding costly failures. This project experiments with time-series modeling techniques (e.g. recurrent neural networks, transformers, or gradient boosting on features) to forecast RUL given sensor histories and operational settings.

Through preprocessing, feature engineering, modeling, and deployment in a small app, this repository demonstrates an end-to-end pipeline for predictive maintenance of turbofan systems.

## Key Features

- Preprocessing pipeline for multivariate time-series sensor data  
- Feature engineering, rolling window aggregation, normalization  
- Model training & evaluation scripts  
- App / interface to upload new time-series and obtain RUL predictions  
- Modular code base so you can swap in different models or datasets  

## Project Structure


.
├── data/                       # raw and processed datasets
│   ├── raw/                    # original dataset files
│   └── processed/              # cleaned / transformed data
├── outputs/                    # model outputs, predictions, plots
├── src/                        # core code (preprocessing, modeling, utils)
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── app.py                       # user interface / API for inference
├── requirements.txt             # required Python packages
└── README.md                    # this documentation

````

Here’s a brief of key files / directories:

- data/raw/ — Raw datasets (e.g. as provided by challenge sources)  
- data/processed/ — Cleaned, normalized, feature-engineered data ready for modeling  
- src/train.py — Training script (load data, train model, log metrics)  
- src/evaluate.py — Evaluate model performance, compute metrics  
- src/inference.py — Inference logic to predict RUL from new data  
- app.py — Simple frontend (e.g. Flask / Streamlit) to accept new input and return RUL  
- outputs/ — Contains model checkpoints, prediction logs, plots, metrics  

## Getting Started

### Prerequisites

- Python 3.7 or above  
- Standard data science / ML libraries (NumPy, pandas, scikit-learn, PyTorch / TensorFlow, etc.)  
- (Optional) GPU for faster training  

### Installation

1. Clone the repository:

      git clone https://github.com/DeepakMallesh/upskillcampus_turbofan.git
   cd upskillcampus_turbofan
````

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # on Windows: venv\Scripts\activate
   

3. Install dependencies:

      pip install -r requirements.txt
   

### Data Preparation
* Place raw dataset files (e.g. from C-MAPSS or other turbofan RUL datasets) into data/raw/.
* Run preprocessing / feature engineering scripts in src/data_utils.py (or a wrapper) to produce cleaned, normalized data in data/processed/.
* Ensure train/validation/test splits are defined (e.g. by engine id, cycles) for reliable evaluation.

### Training & Evaluation

Use src/train.py to train a model. Example:

python src/train.py --config configs/train_config.yaml

After training, use:

python src/evaluate.py --model_path outputs/best_model.pth \
    --test_data data/processed/test.csv

to compute performance metrics (e.g. RMSE, MAE, score used in prognostics challenges).

### Running Inference / Application

To predict RUL for new data:

python src/inference.py --model_path outputs/best_model.pth \
    --input_csv new_input.csv --output predictions.csv

To launch the app:

python app.py

This opens a UI (e.g. localhost) where you can upload or input time-series data and get back a predicted RUL.

## Model & Methods

You may experiment with:

* Recurrent Neural Networks (LSTM / GRU)
* 1D Convolutional Neural Networks
* Transformer / Attention models for time-series
* Traditional ML (e.g. XGBoost, RandomForest) using hand-crafted features
* Ensembling methods

Hyperparameters, loss functions, and architecture details can be stored in a configs/ directory or inside the training script.

## Usage Examples

Example commands:

# Train
python src/train.py --config configs/train.yaml

# Evaluate
python src/evaluate.py --model_path outputs/checkpoint.pth --test_data data/processed/test.csv

# Inference
python src/inference.py --model_path outputs/final_model.pth --input_csv sample_input.csv

# Start app
python app.py

Make sure your input CSV format aligns (i.e. same columns as used during training: sensor readings, cycle numbers, etc.).

## Results & Metrics

You should record and present metrics like:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* Prognostic score (if following standard challenge metric)
* Plots showing predicted vs actual RUL
* Error distributions
* Case studies on specific engine units

Save them under outputs/ along with model checkpoints.

## Future Work / Roadmap

* Increase model robustness (cross validation, regularization)
* Incorporate uncertainty estimates (e.g. Bayesian models, quantile regression)
* Extend to other fleets or datasets
* Deploy as a REST API / cloud service
* Add alerting / decision support integration

## Contributing

Contributions are welcome! Here’s how to get started:

1. Fork the repo
2. Create a feature branch (git checkout -b feature-name)
3. Implement your changes, test, and document
4. Commit and push to your fork
5. Open a Pull Request with clear description

Please follow coding standards, add comments, and include tests where possible.

## License

Specify your license. For example, MIT:

MIT License

© 2025 Deepak Mallesh

Permission is hereby granted, free of charge, to any person obtaining a copy...

## Contact

If you’d like to reach out:

* Author: Deepak Mallesh
* GitHub: [DeepakMallesh](https://github.com/DeepakMallesh)
* Email: [Deepak M](mailto:deepakmallesh2004@gmail.com)

