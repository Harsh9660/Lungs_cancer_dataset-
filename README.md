# Lung Cancer Prediction with Neural Networks

This project demonstrates a complete machine learning pipeline to predict the likelihood of lung cancer based on tabular patient data. It uses a neural network built with TensorFlow and Keras for classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)
- [Model Performance](#model-performance)

## Project Overview

The goal of this project is to classify whether a patient has lung cancer (`YES` or `NO`) based on 15 clinical attributes. The process involves:
1.  Loading and exploring the dataset.
2.  Preprocessing the data by encoding categorical features and scaling numerical features.
3.  Building, training, and evaluating a deep learning model.

## Features

- **Data Preprocessing**: Cleans the dataset by handling duplicates and uses `LabelEncoder` to convert categorical features into a numerical format.
- **Feature Scaling**: Utilizes `StandardScaler` from Scikit-learn to normalize the feature set, which is crucial for optimal neural network performance.
- **Model Architecture**: A Sequential Keras model with `Dense` layers and `Dropout` for regularization to prevent overfitting.
- **Training & Evaluation**: Trains the model and evaluates its performance on a held-out test set, providing key metrics like accuracy, precision, recall, and F1-score.

## Dataset

The model is trained on the `lung_cancer.csv` dataset. This dataset contains patient records with the following columns:

`GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN, LUNG_CANCER`

The target variable is `LUNG_CANCER`.

## Requirements

The project requires Python 3.x and the following libraries. You can install them using pip:

```bash
pip install pandas scikit-learn tensorflow
```

## How to Run

1.  **Clone the repository** or download the files into a local directory.
2.  **Place the dataset**: Ensure the `lung_cancer.csv` file is in the same directory as the Python scripts.
3.  **Execute the training script**: Open your terminal or command prompt, navigate to the project directory, and run the main training file:
    ```bash
    python Train_Testmodel_lungs.py
    ```

## File Structure

- `Train_Testmodel_lungs.py`: The main script that handles data preprocessing, model building, training, and evaluation.
- `Lungs_cancer_detection.py`: A supplementary script for initial data exploration and analysis.
- `lung_cancer.csv`: The dataset file (not included in this repository).

## Model Performance

After running the script, the model's performance on the test set will be printed to the console, including the final accuracy and a detailed classification report.

**Example Output:**
```
--- Evaluating Model ---
Test Accuracy: 0.9516

--- Classification Report ---
              precision    recall  f1-score   support

           0       0.67      0.50      0.57         8
           1       0.96      0.98      0.97        54

    accuracy                           0.92        62
   macro avg       0.81      0.74      0.77        62
weighted avg       0.92      0.92      0.92        62
```
(Note: Actual metrics may vary slightly on each run due to random initialization.)
