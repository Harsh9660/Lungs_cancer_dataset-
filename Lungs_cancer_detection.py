import pandas as pd 
import numpy as np 
import matplotlib as plt
from glob import glob

from sklearn.model_selection import train_test_split 
from sklearn import metrics

from zipfile import ZipFile
import cv2 
import gc
import os

import tensorflow as tf 
from tensorflow import keras
from keras import layers 

import warnings
warnings.filterwarnings('ignore')

def Lungs_cancer_detection:
    path = 'lung-cancer-dataset.zip'

