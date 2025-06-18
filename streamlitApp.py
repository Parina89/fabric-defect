import tensorflow as tf 
import tensorflow_datasets as tfds
import keras 
import numpy as np 
import matplotlib.pyplot as plt


import pandas as pd 
from keras.applications import VGG19,Xception,VGG16
from keras.layers import Dense , Conv2D , MaxPooling2D , Dropout,Flatten,Convolution2D
from keras.models  import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import io

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim



from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


#call vgg model
vgg_model =  VGG19(include_top=True , weights='imagenet')
for models in vgg_model.layers:
models.trainable= False

#converting from functionally model to sequential model
#removing the last 2 alyer to get rid of output layer in VGG16
vgg_model = keras.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
model = keras.Sequential()
for layer in vgg_model.layers:
  model.add(layer)

#add trianbles layers
model.add(Dense(4056, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)




def load_uploaded_image(file):
    return Image.open(file)

st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

uploaded_file_img, camera_file_img = None, None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    elif input_method == "Camera Input" and camera_file_img:
        selected_img = camera_file_img
    else:
        st.warning("Please upload or click an image.")
        selected_img = None

    if selected_img:
        with st.spinner("Analyzing image..."):
            prediction = Anomaly_Detection(selected_img, data_folder)
            st.success(prediction)
