# Renal Disease Classifier

This project is a deep learning solution for classifying renal diseases using MRI scans. It employs state-of-the-art neural networks and tools to build, train, and deploy the model efficiently. The project is designed to assist medical professionals in identifying renal conditions accurately, improving diagnostic workflows.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Workflow](#project-workflow)
7. [Results](#results)
8. [Contributors](#contributors)
9. [License](#license)

---

## Project Overview
The **Renal Disease Classifier** leverages deep learning techniques to analyze MRI scans of kidneys and predict the likelihood of various renal diseases. The project focuses on:
- Automating the disease classification process.
- Providing an interpretable and deployable model for real-world applications.

The data for the project was curated and processed to ensure high-quality training and testing of the deep learning model.

---

## Features
- **Data Management**: Efficient tracking of datasets and models using DVC.
- **Experiment Tracking**: Comprehensive experiment logging with MLflow and Dagshub.
- **Deep Learning Model**: Uses TensorFlow for creating and training convolutional neural networks (CNNs).
- **Web Deployment**: Flask-based REST API for deploying the model.
- **Visualization**: Interactive visualizations with Matplotlib and Seaborn.

---

## Requirements
Below is the list of dependencies required for this project:

```plaintext
tensorflow==2.12.0
pandas
dvc
mlflow==2.2.2
dagshub
notebook
numpy
matplotlib
seaborn
python-box==6.0.2
pyYAML
tqdm
ensure==1.0.2
joblib
types-PyYAML
scipy
Flask
Flask-Cors
gdown
importlib-metadata==6.0.0
-e .


## Installation
# Clone the repository
git clone https://github.com/your-username/renal-disease-classifier.git
cd renal-disease-classifier


# Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

## Usage
# Running the Training Pipeline
1. Set up the DVC pipeline:
- dvc repro

2. Train the model using the defined scripts:
 python src/train.py

## Model Deployment
1. Start the Flask API server:
    python app.py

2. Access the API locally:

Base URL: http://127.0.0.1:5000
Endpoints:
/predict: Accepts MRI image data for predictions.

## Project Workflow
1. Data Collection and Storage:

Data is sourced and managed with DVC.
MRI scans are preprocessed for training.

2. Model Training:

A convolutional neural network (CNN) is trained using TensorFlow.
MLflow tracks experiments, hyperparameters, and model metrics.

3. Evaluation:

The model is evaluated using metrics such as accuracy, precision, and recall.
Visualizations of training performance are created with Matplotlib and Seaborn.

4. Deployment:

A Flask REST API is built to serve predictions.
The API is integrated with Flask-CORS for seamless client interaction.

5. Monitoring:

Use logs and MLflow tracking for ongoing performance monitoring.

## Results
The project achieves:

High accuracy in classifying renal diseases.
Deployment-ready model accessible through a REST API.