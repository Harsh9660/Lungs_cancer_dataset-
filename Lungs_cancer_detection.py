import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf 
from tensorflow import keras
from keras import layers 

import warnings
warnings.filterwarnings('ignore')

def explore_and_preprocess_data(file_path):
    """
    Loads, explores, and preprocesses the lung cancer dataset.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A cleaned and preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    print("--- Initial Data Exploration ---")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nDescriptive Statistics:")
    df.describe()
    print("\nChecking for null values:")
    print(df.isnull().sum())

    
    df.drop_duplicates(inplace=True)
    print(f"\nShape after dropping duplicates: {df.shape}")

   
    if 'GENDER' in df.columns:
        le = LabelEncoder()
        df['GENDER'] = le.fit_transform(df['GENDER'])
        print("\n'GENDER' column encoded.")

    return df

if __name__ == '__main__':
    csv_path = 'lung_cancer.csv' 
    cleaned_df = explore_and_preprocess_data(csv_path)

   
    
    X = cleaned_df.drop('LUNG_CANCER', axis=1)
    y = cleaned_df['LUNG_CANCER']