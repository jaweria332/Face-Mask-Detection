# Importing necessary files
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initializing initial learning rate, number of epochs and batch size
init_lr = 1e-4          #learning rate should be small so that training can be better
epochs = 20
batch_size = 32

# Defining the directotry of dataset and categories present in dataset
directory = "E:\\6th Semester\\AI Practical\\03 Face Mask Detector\\Face-Mask-Detection\\dataset"
categories = ["with_mask", "without_mask"]

print("Loading images...")

# defining variables
data = []
labels = []
for category in categories:
    # Getting path of images, one by one
    path = os.path.join(directory, category)
    #Listdir listdown all the images inside a particular directory
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


#Performing one hot encoding
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#Converting label list into numpy array
data=np.array(data,dtype='float32')
labels=np.array(labels)

X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.20,statify=labels,random_state=42)


#Construct Image data genertaor - create many image from single image
aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

#Laoding MobileNetV2 network, ensuring head FC layer set
baseModel = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

#Construct the head of model that will be place on te top of the base model
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(128,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)

#Place head model on the top of base model
model=Model(inputs=baseModel.input,outputs=headModel)

