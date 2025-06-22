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

# Sidebar info
with st.sidebar:
    img = Image.open("fabric.png")
    st.image(img)
    st.header("About Project")
    st.write("This app detects defects like stains on fabrics made by power looms. Quality control via AI-based image classification.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
class FabricDefectClassifier(nn.Module):
    def __init__(self):
        super(FabricDefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: Defect-Free, Stain

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model function
@st.cache(allow_output_mutation=True)
def load_model():
    model = FabricDefectClassifier()
    model_path = "textile.pth"  # Make sure this file exists in the working dir
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load only state_dict
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction
def get_prediction(image):
    image = transform_image(image).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_labels = ['Defect-Free', 'Stain']  # Adjust as per training

# Handling Upload
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
