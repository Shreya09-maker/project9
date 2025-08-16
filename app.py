import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Google Drive file ID of your tomato_model.h5
FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
MODEL_PATH = "tomato_model.h5"

# Download and load the model (only once per session)
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# Load model
model = download_and_load_model()

# Class names from your trained model
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

def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx] * 100

def is_tomato_leaf_color(image: Image.Image):
    img_np = np.array(image.convert('RGB'))
    lower = np.array([20, 60, 20])
    upper = np.array([100, 255, 100])
    mask = ((img_np[:, :, 0] >= lower[0]) & (img_np[:, :, 0] <= upper[0]) &
            (img_np[:, :, 1] >= lower[1]) & (img_np[:, :, 1] <= upper[1]) &
            (img_np[:, :, 2] >= lower[2]) & (img_np[:, :, 2] <= upper[2]))
    green_ratio = np.sum(mask) / (img_np.shape[0] * img_np.shape[1])
    return green_ratio > 0.12

# Streamlit UI setup
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")

with st.sidebar:
    st.header("About")
    st.write("""
        Upload a tomato leaf image to detect diseases using a deep learning model.
        
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
    st.write("Developed by Shreya Patil 🍅")

st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a tomato leaf image to begin diagnosis.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    if not is_tomato_leaf_color(image):
        st.error("❌ Not enough green pixels — please upload a valid tomato leaf image.")
    else:
        label, confidence = predict(image)
        if confidence < 60:
            st.error("❌ Low confidence — try another image.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Tomato Leaf", use_container_width=True)
            with col2:
                st.markdown(f"<h3>Prediction:</h3><h2 style='color:red'>{label.replace('_',' ')}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3>Confidence:</h3>")
                st.progress(min(int(confidence), 100))
                st.markdown(f"<h4>{confidence:.2f}%</h4>", unsafe_allow_html=True)
else:
    st.info("Please upload an image file to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Tomato Leaf Disease Detector</p>", unsafe_allow_html=True)
