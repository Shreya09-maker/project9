import os
os.system("pip install matplotlib")

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Load your trained tomato leaf disease model
MODEL_PATH = r"C:\Users\shrey\Downloads\New folder\tomato_model.h5"
model = load_model(MODEL_PATH)

# Class names corresponding to your model outputs
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Optional: simple remedies/info for diseases
disease_info = {
    "Tomato___Bacterial_spot": "Spray copper-based fungicides. Avoid overhead watering.",
    "Tomato___Early_blight": "Use fungicides like chlorothalonil. Rotate crops.",
    "Tomato___Late_blight": "Remove affected plants. Apply fungicide sprays.",
    "Tomato___Leaf_Mold": "Improve air circulation. Apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Use preventive fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides. Spray neem oil.",
    "Tomato___Target_Spot": "Use fungicides like mancozeb. Remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants. Control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Destroy infected plants. Disinfect tools.",
    "Tomato___healthy": "Your plant is healthy 🍀. Keep monitoring regularly."
}

# ---------------- Utility Functions ----------------
def preprocess_image(image: Image.Image):
    """Resize and scale image for model input."""
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    """Predict disease label and confidence from image."""
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    predicted_index = np.argmax(preds)
    confidence = preds[predicted_index] * 100
    predicted_label = class_names[predicted_index]
    return predicted_label, confidence, preds

def is_tomato_leaf_color(image: Image.Image):
    """Check if image has enough green pixels to be considered a tomato leaf."""
    img_np = np.array(image.convert('RGB'))
    
    # Stricter green pixel thresholds
    lower = np.array([20, 60, 20])
    upper = np.array([100, 255, 100])
    
    mask = ((img_np[:, :, 0] >= lower[0]) & (img_np[:, :, 0] <= upper[0]) &
            (img_np[:, :, 1] >= lower[1]) & (img_np[:, :, 1] <= upper[1]) &
            (img_np[:, :, 2] >= lower[2]) & (img_np[:, :, 2] <= upper[2]))
    
    green_ratio = np.sum(mask) / (img_np.shape[0] * img_np.shape[1])
    return green_ratio > 0.12  # Require at least 12% green pixels

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")

with st.sidebar:
    st.header("About")
    st.write("""
        This app detects **tomato leaf diseases** using a deep learning model.  
        Upload an image of a tomato leaf to get the disease prediction and confidence score.  
        
        **Class Labels:**
        - Bacterial Spot
        - Early Blight
        - Late Blight
        - Leaf Mold
        - Septoria Leaf Spot
        - Spider Mites (Two-spotted)
        - Target Spot
        - Tomato Yellow Leaf Curl Virus
        - Tomato Mosaic Virus
        - Healthy
    """)
    st.markdown("---")
    st.write("Developed by **Shreya Patil** 🍅")

st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image of a tomato leaf below to get started.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Check green pixel ratio first
    if not is_tomato_leaf_color(image):
        st.error("❌ This does not look like a tomato leaf (not enough green pixels). Please upload a valid tomato leaf image.")
    else:
        predicted_label, confidence, preds = predict(image)
        confidence_threshold = 60  # Adjust as needed

        if (predicted_label not in class_names) or (confidence < confidence_threshold):
            st.error("❌ The uploaded image does not appear to be a tomato leaf from the dataset. Please upload a valid tomato leaf image.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded Tomato Leaf Image", use_container_width=True)
            with col2:
                st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: #d32f2f;'>{predicted_label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #4CAF50;'>Confidence:</h3>", unsafe_allow_html=True)
                st.progress(min(int(confidence), 100))
                st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)

                # Show remedy info
                if predicted_label in disease_info:
                    st.info(f"💡 **Recommendation:** {disease_info[predicted_label]}")

            # Show probability distribution as bar chart
            st.subheader("Prediction Probability for Each Class")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(class_names, preds * 100)
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Model Prediction Distribution")
            st.pyplot(fig)

else:
    st.info("📌 Please upload an image file to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Tomato Leaf Disease Detector | Powered by TensorFlow & Streamlit 🍅</p>", unsafe_allow_html=True)

