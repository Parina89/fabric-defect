import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
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

# Load model function
@st.cache_resource()  # updated streamlit cache
def load_model():
    model = FabricDefectClassifier()
    model_path = r"textile.pth"  # Make sure textile.pth is in the same folder
    state_dict = torch.load(model_path, map_location=device)  # loading weights only
    model.load_state_dict(state_dict)  # load into the defined model
    model.eval()  # set to eval mode
    return model

model = load_model()

# Image Transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)  # Removed extra unsqueeze here

# Prediction
def get_prediction(image):
    image = transform_image(image).unsqueeze(0)  # Add batch dimension here
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_labels = ['Defect-Free', 'Stain']

# Handle file upload
uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"])

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
