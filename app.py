import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Google Drive File ID and Model Path
FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
MODEL_PATH = "tomato_model.h5"

# Download the model if it's not already present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# Load model after ensuring it's downloaded
model = download_and_load_model()

# Disease class labels
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
    green_ratio = np.sum(mask)_
