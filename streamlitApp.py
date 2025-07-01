import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Image preprocessing function
def transform_image(image):
    """
    Transform input image for model prediction
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Handle different numpy array formats
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            # Convert to uint8 if needed
            image = Image.fromarray((image * 255).astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image or numpy array")
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def get_prediction(image, model, threshold=0.5, debug=True):
    """
    Predict whether image contains stains or is defect-free
    
    Args:
        image: PIL Image or numpy array
        model: Trained PyTorch model
        threshold: Classification threshold (default: 0.5)
        debug: Print debug information
    
    Returns:
        tuple: (prediction_string, probability_score)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        try:
            # Transform image for model input
            image_tensor = transform_image(image)
            
            # Move to same device as model
            device = next(model.parameters()).device
            image_tensor = image_tensor.to(device)
            
            # Get model output
            outputs = model(image_tensor)
            
            # Handle different output formats
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-class output - use softmax
                probs = torch.softmax(outputs, dim=1)
                prob_score = probs[0, 1].item()  # Probability of stain class
            else:
                # Binary output - use sigmoid
                probs = torch.sigmoid(outputs)
                prob_score = probs.item() if probs.dim() == 0 else probs[0].item()
            
            if debug:
                print(f"Raw model output: {outputs}")
                print(f"Processed probability: {prob_score}")
                print(f"Threshold: {threshold}")
            
            # CORRECTED LOGIC: 
            # Higher probability = more likely to be stain
            # Lower probability = more likely to be defect-free
            if prob_score > threshold:
                prediction = "stain"
            else:
                prediction = "defect-free"
            
            if debug:
                print(f"Final prediction: {prediction}")
            
            return prediction, prob_score
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0

# Alternative function with inverted logic (if your model is trained differently)
def get_prediction_inverted(image, model, threshold=0.5, debug=True):
    """
    Alternative prediction function with inverted logic
    Use this if your model outputs higher probabilities for defect-free images
    """
    model.eval()
    
    with torch.no_grad():
        try:
            image_tensor = transform_image(image)
            device = next(model.parameters()).device
            image_tensor = image_tensor.to(device)
            
            outputs = model(image_tensor)
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.softmax(outputs, dim=1)
                prob_score = probs[0, 0].item()  # Probability of defect-free class
            else:
                probs = torch.sigmoid(outputs)
                prob_score = probs.item() if probs.dim() == 0 else probs[0].item()
            
            if debug:
                print(f"Raw model output: {outputs}")
                print(f"Processed probability: {prob_score}")
                print(f"Threshold: {threshold}")
            
            # INVERTED LOGIC:
            # Higher probability = more likely to be defect-free
            # Lower probability = more likely to be stain
            if prob_score > threshold:
                prediction = "defect-free"
            else:
                prediction = "stain"
            
            if debug:
                print(f"Final prediction: {prediction}")
            
            return prediction, prob_score
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0

def predict_from_path(image_path, model, use_inverted=False, threshold=0.5):
    """
    Load image from path and make prediction with error handling
    """
    try:
        # Load and validate image
        image = Image.open(image_path)
        
        # Choose prediction function
        if use_inverted:
            result, prob = get_prediction_inverted(image, model, threshold, debug=True)
        else:
            result, prob = get_prediction(image, model, threshold, debug=True)
        
        print(f"\nImage: {image_path}")
        print(f"Prediction: {result}")
        print(f"Confidence: {prob:.4f}")
        print("-" * 40)
        
        return result, prob
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "error", 0.0

def calibrate_threshold(model, test_images_with_labels, use_inverted=False):
    """
    Find optimal threshold using test images
    
    Args:
        model: Trained model
        test_images_with_labels: List of tuples (image_path, true_label)
        use_inverted: Whether to use inverted prediction logic
    
    Returns:
        float: Optimal threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_threshold = 0.5
    best_accuracy = 0.0
    
    for threshold in thresholds:
        correct = 0
        total = 0
        
        for image_path, true_label in test_images_with_labels:
            try:
                image = Image.open(image_path)
                if use_inverted:
                    pred, _ = get_prediction_inverted(image, model, threshold, debug=False)
                else:
                    pred, _ = get_prediction(image, model, threshold, debug=False)
                
                if pred == true_label:
                    correct += 1
                total += 1
            except:
                continue
        
        if total > 0:
            accuracy = correct / total
            print(f"Threshold {threshold:.1f}: Accuracy = {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold} (Accuracy: {best_accuracy:.3f})")
    return best_threshold

# Example model definition
class StainDetectionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(StainDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Example usage and testing
if __name__ == "__main__":
    # Load your trained model
    model = StainDetectionModel()
    # model.load_state_dict(torch.load('your_model.pth'))
    
    # Test with different approaches
    print("Testing prediction functions...")
    
    # Method 1: Standard prediction
    # result1, prob1 = predict_from_path('defect_free_image.jpg', model, use_inverted=False)
    
    # Method 2: Inverted prediction (try this if Method 1 gives wrong results)
    # result2, prob2 = predict_from_path('defect_free_image.jpg', model, use_inverted=True)
    
    # Method 3: Custom threshold
    # result3, prob3 = predict_from_path('defect_free_image.jpg', model, threshold=0.3)
    
    print("Choose the method that gives correct results for your model!")
    print("If defect-free images are still showing as 'stain', try:")
    print("1. use_inverted=True")
    print("2. Lower threshold (e.g., 0.3)")
    print("3. Use calibrate_threshold() function")
