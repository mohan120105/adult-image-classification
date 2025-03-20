import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests
from PIL import Image

# Define the Google Drive link for the model file
GOOGLE_DRIVE_LINK = 'https://drive.google.com/uc?id=1n3-mQzW4urQW-xypoNgky6bjy33NXeTp'
MODEL_PATH = 'nsfw_classifier.h5'

# Download the model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait.")
        response = requests.get(GOOGLE_DRIVE_LINK, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            st.success("Model downloaded successfully.")
        else:
            st.error("Failed to download model.")

# Load the trained model
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error("Model file not found. Ensure the model is downloaded correctly.")
        return None

# Initialize the model
download_model()
model = load_trained_model()

# Preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI
st.title("🔍 NSFW Image Classifier")
st.write("Upload an image to classify whether it contains adult content or is safe.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Image"):
        if model:
            img_array = preprocess_image(image_display)
            prediction = model.predict(img_array)
            result = "🚨 Adult Content" if prediction[0][0] > 0.5 else "✅ Safe Content"
            st.subheader("Prediction:")
            st.write(result)
        else:
            st.error("Model not loaded correctly. Please try again.")
