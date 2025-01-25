from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the Google Drive link for the model file
GOOGLE_DRIVE_LINK = 'https://drive.google.com/file/d/1n3-mQzW4urQW-xypoNgky6bjy33NXeTp/view?usp=sharing'
MODEL_PATH = 'nsfw_classifier.h5'

# Download the model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(GOOGLE_DRIVE_LINK, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print("Model downloaded successfully.")
        else:
            print("Failed to download model.")

# Load the trained model
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("Model file not found. Ensure the model is downloaded correctly.")

# Initialize the model
print("Initializing the application...")
download_model()
model = load_trained_model()

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index2.html')

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        result = 'Adult Content' if prediction[0][0] > 0.5 else 'Safe Content'

        # Clean up the uploaded file
        os.remove(file_path)

        return jsonify({'prediction': result})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
