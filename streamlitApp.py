import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Fabric Inspector", page_icon="🧵")
st.title("Fabric Defect Classification on Power Looms")
st.write("Upload an image of fabric and the AI model will classify it as 'Defect-Free' or 'Stain'.")

# Sidebar information
with st.sidebar:
    # Check if fabric.png exists, otherwise show a placeholder
    fabric_image_path = "fabric.png"
    if os.path.exists(fabric_image_path):
        img = Image.open(fabric_image_path)
        st.image(img)
    else:
        st.info("Add fabric.png to display sidebar image")
    
    st.header("About Project")
    st.write("This app helps detect defects like stains on fabrics produced by power looms. It ensures quality control in real-time production environments using AI-based image classification.")

uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (Simple CNN)
class FabricDefectClassifier(nn.Module):
    def __init__(self):  # Fixed __init__ method
        super(FabricDefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: defect-free, stain
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model with proper error handling
@st.cache_resource  # Updated cache decorator
def load_model():
    try:
        model = FabricDefectClassifier()
        model_path = "textile.pth"  # Make sure this file exists
        
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct location.")
            return None
            
        # Fixed model loading
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image Transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_prediction(image, model):
    if model is None:
        return "Error: Model not loaded"
    
    try:
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():  # Disable gradient computation
            # Transform image
            image_tensor = transform_image(image).to(device)
            
            # Get model output
            outputs = model(image_tensor)
            
            # Apply softmax to get probabilities for 2-class output
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy().squeeze()
            
            # Debug information (optional - can be removed in production)
            st.write(f"Model probabilities: Defect-free: {probs[0]:.4f}, Stain: {probs[1]:.4f}")
            
            # Get prediction based on highest probability
            predicted_class = np.argmax(probs)
            
            # Map to class labels
            class_labels = ['defect-free', 'stain']
            label = class_labels[predicted_class]
            confidence = probs[predicted_class]
            
            return label, confidence
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Error", 0.0

# Load the model
model = load_model()

# Main app logic
if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Fabric Image', use_column_width=True)
        
        if model is not None:
            st.write("Classifying...")
            
            # Get prediction
            result = get_prediction(image, model)
            
            if isinstance(result, tuple):
                prediction, confidence = result
                
                # Display results
                st.success(f"**Prediction: {prediction.upper()}**")
                st.info(f"**Confidence: {confidence:.2%}**")
                
                # Provide appropriate feedback
                if prediction == 'defect-free':
                    st.success("✅ The fabric appears to be free of defects.")
                else:
                    st.error("⚠️ Stain detected! Please check this fabric.")
                    
                # Add confidence-based warnings
                if confidence < 0.7:
                    st.warning("⚡ Low confidence prediction. Consider manual inspection.")
                    
            else:
                st.error("Error in prediction process.")
        else:
            st.error("Model could not be loaded. Please check if 'textile.pth' exists and is valid.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.info("👆 Please upload a fabric image to get started.")
    
    # Optional: Show example or instructions
    st.markdown("""
    ### Instructions:
    1. Upload a clear image of fabric
    2. Supported formats: JPG, JPEG, PNG
    3. The AI will analyze the image and detect stains or defects
    4. Results will show the prediction with confidence score
    """)

# Footer
st.markdown("---")
st.markdown("Made with Streamlit 🎈")
