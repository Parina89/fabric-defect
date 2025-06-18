import streamlit as st
import io

#import torch
#import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
#import torch
#import torch.nn as nn
#import torch.optim as optim



from PIL import Image
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader, Dataset




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


