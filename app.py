import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Parameters
img_size = 224
categories = ['Cephalometric', 'Anteroposterior', 'OPG']

# Function to prepare image (for prediction)
def prepare_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (img_size, img_size))
    normalized_array = resized_array / 255.0
    return normalized_array.reshape(-1, img_size, img_size, 1)

# Function to predict image
def predict_image(image_path, model):
    prepared_image = prepare_image(image_path)
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return categories[predicted_class]

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

model = load_model()

# Streamlit app
st.title("X-Ray Image Classification")

uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    results = []
    temp_dir = "temp_dir"

    # Create temp_dir if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict the class of the image
        prediction = predict_image(temp_file_path, model)
        results.append((uploaded_file.name, prediction))

    # Display results in a table
    st.write("## Predictions")
    if results:
        table_data = [(result[0], result[1]) for result in results]
        st.table(table_data)
