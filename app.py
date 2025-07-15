import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

st.title("♻️ Smart Waste Classifier")
st.write("Upload an image to classify it as **Organic** or **Recyclable**.")

# Load model
try:
    model = tf.keras.models.load_model("waste_classifier.h5")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

class_names = ['Organic', 'Recyclable']

# File uploader — this should always show
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

# If user uploads image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Prediction: **{predicted_class}**")
