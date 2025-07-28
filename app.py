import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model_trained.h5")

# Constants
image_size = (32, 32)
label_file = "labels.csv"  # Your CSV with class names
labels = pd.read_csv(label_file)
class_names = labels["Name"].tolist()

# Preprocessing function (same as used during training)
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img.reshape(32, 32, 1)

# Streamlit UI
st.title("Traffic Sign/Class Image Classification")
st.write("Upload an image to classify it using the trained CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize(image_size)
    image = np.array(image)
    try:
        processed_image = preprocess_image(image)
        processed_image = processed_image.reshape(1, 32, 32, 1)

        # Predict
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Prediction: **{class_names[predicted_class]}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")
    except Exception as e:
        st.error(f"Error in processing the image: {str(e)}")
