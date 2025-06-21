import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import urllib.request

# Set page config
st.set_page_config(page_title="Fabric Inspector", page_icon="🧵")
st.title("Fabric Defect Classification on Power Looms")
st.write("Upload an image of fabric and the AI model will classify it as 'Defect-Free' or 'Stain'.")

# Sidebar information
with st.sidebar:
    img = Image.open(r"fabric.png")
    st.image(img)
    st.header("About Project")
    st.write("This app helps detect defects like stains on fabrics produced by power looms. It ensures quality control in real-time production environments using AI-based image classification.")

# Image Upload
uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (Simple Transfer Learning)
class FabricDefectClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FabricDefectClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Using ResNet18
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # 2 classes: Defect-Free, Stain

    def forward(self, x):
        return self.model(x)

# Load model (Assuming you have trained model stored as 'fabric_model.pth')
@st.cache(allow_output_mutation=True)
def load_model():
    model = FabricDefectClassifier()
    #url = 'https://drive.google.com/file/d/1rWsqzW6UIL5pxjczNH72d3_3WUl27Rza/view?usp=drive_link'
    #urllib.request.urlretrieve(url, 'textile.h5')
    model_path = "textile.pth"  # <-- Update path
    model.load_state_dict(torch.load("textile.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image Transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction
def get_prediction(image):
    image_tensor = transform_image(image).to(device)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Class labels
class_labels = ['Defect-Free', 'Stain']

# Handle image prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Fabric Image', use_column_width=True)
    st.write("Classifying...")

    prediction = get_prediction(image)
    result = class_labels[prediction]

    st.success(f"Prediction: **{result}**")
    if result == 'Defect-Free':
        st.info("The fabric appears to be free of defects.")
    else:
        st.warning("Stain detected! Please check this fabric.")

