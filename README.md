# Crop & Weed Detection

A computer vision / deep learning project to detect and differentiate between crops and weeds in agricultural fields. This tool aims to help automate weed control, reduce herbicide usage, and support precision farming.

## 📘 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)  
- [Model Training & Inference](#model-training--inference)  
- [Streamlit App](#streamlit-app)  
- [Datasets](#datasets)  
- [Results & Outputs](#results--outputs)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)

---

## Overview

Agricultural fields are often invaded by weeds, which reduce crop yield and increase maintenance cost. This project uses object detection methods to identify crops and weeds in images captured from fields, enabling automated or semi-automated weed removal.  

The repository includes data processing, training, inference scripts, and a simple web app interface (via Streamlit) for demonstration.

## Features

- Trainable object detection model (e.g. using YOLO or similar)  
- Inference on new images to highlight weeds vs crops  
- Web UI for quick testing  
- Modular scripts to adapt to different datasets or models  

## Project Structure


.
├── data/                     # raw and processed datasets
├── models/                   # saved model weights & architectures
├── outputs/                  # inference results, predictions, visualizations
├── scripts/                  # helper scripts (e.g. preprocessing, utilities)
├── streamlit_app.py          # front-end web app
├── train_and_infer.py        # training & inference orchestration
├── requirements.txt          # Python dependencies
└── yolov5su.pt               # pretrained or baseline model weights

````

- data/ — Contains your training, validation, and test image sets and annotation files.  
- models/ — Stores trained model weights, checkpoints, model definitions.  
- outputs/ — Where the generated prediction images, logs, and metrics are saved.  
- scripts/ — Utility scripts (e.g. annotation parsing, image augmentations).  
- streamlit_app.py — Runs a web front-end to upload images and display predictions.  
- train_and_infer.py — Core script to train the model or run inference.  
- requirements.txt — Lists all necessary Python packages.  
- yolov5su.pt — A default / baseline model weight file included (you may replace or retrain it).

## Getting Started

### Prerequisites

- Python 3.7+  
- GPU (CUDA-enabled) is recommended for training / faster inference  
- pip (or conda) to install packages  

### Installation

1. Clone the repository:

      git clone https://github.com/DeepakMallesh/upskillcampus_Crop-and-Weed-Detection.git
   cd upskillcampus_Crop-and-Weed-Detection
````

2. Create & activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   

3. Install required packages:

      pip install -r requirements.txt
   

### Usage

There are two main modes:

* Training / inference via CLI
* Web interface via Streamlit

#### Command-Line Usage (train & inference)

python train_and_infer.py --mode train --config path/to/config.yaml
python train_and_infer.py --mode infer --input path/to/images --output outputs/

You can customize config files (if supported) to set hyperparameters, augmentation, paths, etc.

#### Running the Streamlit App

streamlit run streamlit_app.py

Then open the provided local URL (e.g. http://localhost:8501) in your browser. You’ll see a simple interface to upload an image and get predictions.

## Model Training & Inference
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
