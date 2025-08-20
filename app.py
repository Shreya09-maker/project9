import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import gdown

# --------------- Configuration ---------------

DRIVE_FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
MODEL_PATH = "tomato_model.h5"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

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

# --------------- Model Download & Load ---------------

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
    st.error("Model file is missing or invalid. Please check model upload.")
    st.stop()

model = load_model(MODEL_PATH)

# --------------- Prediction & Preprocessing ---------------

def preprocess_image(image: Image.Image):
    img = image.convert('RGB').resize((150, 150))
    arr = img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image: Image.Image):
    arr = preprocess_image(image)
    preds = model.predict(arr)[0]
    return [(class_names[i], preds[i] * 100) for i in preds.argsort()[::-1]]

def is_tomato_leaf(image: Image.Image, threshold=0.10):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    green_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
    return green_ratio > threshold

# --------------- Streamlit UI ---------------

st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide")
st.title("🍅 Tomato Leaf Disease Detector")

uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    if not is_tomato_leaf(img):
        st.warning("⚠️ Uploaded image may not be a tomato leaf. Predictions may not be accurate.")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    predictions = predict(img)
    confident_preds = [(label, conf) for label, conf in predictions if conf >= 90]

    if confident_preds:
        st.success("Here are the predictions with >90% confidence:")
        for label, conf in confident_preds:
            label_text = label.replace('_', ' ')
            st.write(f"**{label_text}**: {conf:.2f}%")
            st.progress(min(int(conf), 100))
    else:
        st.info("No predictions reached 90% confidence. Here are the top candidates:")
        top_label, top_conf = predictions[0]
        st.write(f"- **{top_label.replace('_', ' ')}**: {top_conf:.2f}%")
        st.progress(min(int(top_conf), 100))
else:
    st.info("Please upload an image to get started.")
