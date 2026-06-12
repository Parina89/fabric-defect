# Fabric Defect Detection

A deep learning–based web app that automatically detects defects in fabric/textile images, helping automate visual quality control in textile manufacturing.

## 🧵 Problem Statement

Manual fabric inspection is slow, subjective, and prone to human error. This project uses a trained PyTorch model to classify fabric images as **defective** or **non-defective**, enabling faster and more consistent quality checks.


## 🛠️ Tech Stack

- **Python**
- **PyTorch** — model training & inference (`textile.pth`)
- **Streamlit** — interactive web interface (`streamlitApp.py`)
- **Jupyter Notebook** — model training & experimentation

## 📂 Project Structure

```
fabric-defect/
├── streamlitApp.py      # Streamlit web app for inference
├── textile.pth          # Trained PyTorch model weights
├── Untitled5.ipynb       # Model training/experimentation notebook
├── requirements.txt      # Python dependencies
└── fabric.png            # Sample fabric image
```

## ⚙️ How It Works

1. User uploads a fabric image through the Streamlit interface.
2. The image is preprocessed and passed through the trained PyTorch model (`textile.pth`).
3. The model predicts whether the fabric sample is **defective** or **non-defective**.
4. The result is displayed instantly in the web app.

## 💻 Running Locally

```bash
# Clone the repository
git clone https://github.com/Parina89/fabric-defect.git
cd fabric-defect

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlitApp.py
```

## 👩‍💻 Author

**Parina Vaghadia**
[LinkedIn](https://www.linkedin.com/in/parina-vaghadia-a31360232) · [GitHub](https://github.com/Parina89)
