import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

st.set_page_config(page_title="Fabric Inspector", page_icon="🧵")

st.title("Fabric Defect Classification on Power Looms")

# Sidebar
with st.sidebar:
    img = Image.open("fabric.png")
    st.image(img)
    st.header("About Project")
    st.write("This app detects stains on fabrics using AI.")

uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"], key="main_uploader")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
    model.load_state_dict(torch.load("textile.pth", map_location=device))
    model.eval()
    return model

model = load_model()

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image):
    image = transform_image(image).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

class_labels = ['Defect-Free', 'Stain']

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
