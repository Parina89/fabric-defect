import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np

# Set page config
st.set_page_config(page_title="Fabric Inspector", page_icon="🧵")
st.title("Fabric Defect Classification on Power Looms")
st.write("Upload an image of fabric and the AI model will classify it as 'Defect-Free' or 'Stain'.")
st.markdown("""
<style>
.main {
    background-color: #1E90FF;
}
</style>
""", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    img = Image.open(r"fabric.png")
    st.image(img)
    st.header("About Project")
    st.write("This app helps detect defects like stains on fabrics produced by power looms. It ensures quality control in real-time production environments using AI-based image classification.")
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]  {
    background-color: #191970;
    border-right: 2px solid #2C3E50;
    }
    </style>
   """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (Simple Transfer Learning)
class FabricDefectClassifier(nn.Module):
    def __init__(self):
        super(FabricDefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model (Assuming you have trained model stored as 'fabric_model.pth')
@st.cache(allow_output_mutation=True)
def load_model():
    model = FabricDefectClassifier()
    model_path = r"textile.pth"  # <-- Update path
    model.load_state_dict = torch.load(model_path, map_location=device)
    #model.eval()
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

def get_prediction(image):
    image = transform_image(image)  # Add batch dimension
    outputs = model(image)          # Get model output (logits or probabilities)
    _, predicted = torch.max(outputs, 1)  # Pick the class with highest score
    probs = torch.sigmoid(outputs).detach().cpu().numpy().squeeze()
    #print("Model probabilities:" , probs)  # Temporarily add this for debugging
    #label = "defect-free" if probs[0] <= 0.5 else "stain"
    if len(probs.shape) == 0:  # Single output (binary classification)
        label = "stain" if probs > 0.5 else "defect-free"
    else:  # Multiple outputs
        label = "stain" if probs[0] > 0.5 else "defect-free"
    return label
    
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Fabric Image', use_column_width=True)
    st.write("Classifying...")
    
    class_labels = ['defect-free','stain'] # adjust as per your training labels 
    prediction = get_prediction(image)
    result = prediction  # already the label


    st.success(f"Prediction: **{result}**")
    
    if result == 'defect-free':
        st.success("The fabric appears to be free of defects.")
    else:
        st.error("Stain detected! Please check this fabric.")
