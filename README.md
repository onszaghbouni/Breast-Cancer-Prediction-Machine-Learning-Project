# Breast-Cancer-Prediction-Machine-Learning-Project
📌 Overview

This project aims to build a machine learning model that predicts whether a breast tumor is benign or malignant using the Breast Cancer Wisconsin Dataset.
It includes data preprocessing, exploratory data analysis (EDA), visualizations, model training, and a prediction application.

The trained model (best_model.pkl) and scaler (scaler.pkl) are included to allow direct inference without retraining.

📁 Project Structure
ML-Breast-Cancer-Project/
│
├── app.py                   # Prediction application (loads model & scaler)
├── model_training.py        # Script used to train the ML model
├── cancer_visualization.py  # Generates dataset visualizations
│
├── models/
│   ├── best_model.pkl       # Saved trained ML model
│   └── scaler.pkl           # Preprocessing scaler
│
├── plots/                   # Visualizations generated during EDA
│   ├── plot_cancer_class_distribution.png
│   ├── plot_cancer_heatmap.png
│   └── plot_cancer_scatter.png
│
├── requirements.txt         # Python dependencies
└── README.md                # Documentation

🎯 Project Objectives

Understand and clean the Breast Cancer dataset

Perform exploratory data analysis and visualize key relationships

Train and optimize an ML model for cancer classification

Save the best model for later use

Develop a small application (app.py) that predicts a diagnosis from new data

📊 Visualizations Included

The repository contains multiple EDA plots:

Class Distribution Plot

Correlation Heatmap

Feature Scatter Plot

These visuals help understand feature relationships and dataset patterns.

🧠 Model Information

Dataset: Breast Cancer Wisconsin (from scikit-learn)

Preprocessing: StandardScaler

Model Training Script: model_training.py

Outputs:

best_model.pkl → Trained model

scaler.pkl → Feature scaler

Metrics evaluated: Accuracy, confusion matrix, classification performance (depending on your code)

▶️ How to Run
1. Clone the Repository
git clone https://github.com/YourUsername/ML-Breast-Cancer-Project.git
cd ML-Breast-Cancer-Project

2. Install Dependencies
pip install -r requirements.txt

3. (Optional) Retrain the Model
python model_training.py

4. Run the Prediction App
python app.py

🔍 Files Description
✔ app.py

Loads the trained model + scaler and performs predictions for new samples.

✔ model_training.py

Handles model training, preprocessing, and saving the final .pkl files.

✔ cancer_visualization.py

Generates visualizations such as heatmaps and scatter plots.

✔ plots/ Folder

Contains pre-generated EDA images.

✔ models/ Folder

Contains serialized model and scaler.

📦 Requirements

A requirements.txt file is provided.
Install the dependencies before running the project.

📝 Author

This project was developed as part of a machine learning mini-project to demonstrate the full workflow from data processing to deployment-ready prediction.

⭐ If you find this project useful, feel free to give it a star!
