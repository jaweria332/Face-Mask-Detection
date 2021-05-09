#Importing necessary files
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#Initializing initial learning rate, number of epochs and batch size
init_lr = 1e-4
epochs = 20
batch_size = 32

#Defining the directotry of dataset and categories present in dataset
directory = "E:\\6th Semester\\AI Practical\\03 Face Mask Detector\\Face-Mask-Detection\\dataset"
categories = ["with_mask","without_mask"]

