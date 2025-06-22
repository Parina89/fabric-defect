import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Page config
st.set_page_config(page_title="Fabric Inspector", page_icon="🧵")
st.title("Fabric Defect Classification on Power Looms")
st.write("Upload an image of fabric and the AI model will classify it as 'Defect-Free' or 'Stain'.")

# Sidebar info
with st.sidebar:
    img = Image.open("fabric.png")
    st.image(img)
    st.header("About Project")
    st.write("This app detects defects like stains on fabrics made by power looms.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (must match training model)
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

# Load model correctly with state_dict
@st.cache(allow_output_mutation=True)
def load_model():
    model = FabricDefectClassifier()
    model.load_state_dict(torch.load("textile.pth", map_location=device)) # load state dict properly
    model.eval()
    return model

model = load_model()

# Image Preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def get_prediction(image):
    image = transform_image(image).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_labels = ['Defect-Free', 'Stain']

# Handle image upload
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
