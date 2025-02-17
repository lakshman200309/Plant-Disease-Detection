from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = "model/inceptionv3_plant_disease_fixed.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found! Please check the file path.")
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Modify as per your dataset)
class_labels = ['Healthy', 'Powdery Mildew', 'Rust', 'Leaf Spot', 'Blight']

# Load disease information from JSON file
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # Retrieve disease information
    disease_details = disease_info.get(predicted_class, {
        "cure": "No information available.",
        "growth_tips": "No information available."
    })
    
    return render_template('result.html', 
                           prediction=predicted_class, 
                           cure=disease_details["cure"], 
                           growth_tips=disease_details["growth_tips"])

if __name__ == '__main__':
    app.run(debug=True)
