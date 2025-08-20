import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os
import pandas as pd

# ---------------- Download model from Google Drive ----------------
MODEL_PATH = "tomato_model.h5"
FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# ---------------- Class names & disease info ----------------
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

# ---------------- Utility functions ----------------
def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    label = class_names[idx]
    confidence = preds[idx] * 100
    return label, confidence, preds

def is_tomato_leaf_color(image: Image.Image):
    img_np = np.array(image.convert('RGB'))
    lower = np.array([20, 60, 20])
    upper = np.array([100, 255, 100])
    mask = (
        (img_np[..., 0] >= lower[0]) & (img_np[..., 0] <= upper[0]) &
        (img_np[..., 1] >= lower[1]) & (img_np[..., 1] <= upper[1]) &
        (img_np[..., 2] >= lower[2]) & (img_np[..., 2] <= upper[2])
    )
    green_ratio = mask.sum() / (img_np.shape[0] * img_np.shape[1])
    return green_ratio > 0.12

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")
st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    if not is_tomato_leaf_color(image):
        st.error("❌ This does not look like a tomato leaf. Please upload a valid tomato leaf image.")
    else:
        label, confidence, preds = predict(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Tomato Leaf", use_container_width=True)
        with col2:
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #d32f2f;'>{label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)

            if label in disease_info:
                st.info(f"💡 **Recommendation:** {disease_info[label]}")

        # Probability distribution chart
        st.subheader("Prediction Probabilities (%)")
        df = pd.DataFrame({
            "Class": [cn.replace('Tomato___', '').replace('_', ' ') for cn in class_names],
            "Confidence": (preds * 100).round(2)
        }).set_index("Class")
        st.bar_chart(df)

else:
    st.info("📌 Please upload an image to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Tomato Leaf Disease Detector | Powered by TensorFlow & Streamlit 🍅</p>", unsafe_allow_html=True)
