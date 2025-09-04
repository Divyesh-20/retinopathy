import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Path to your trained model (update if needed)
MODEL_PATH = r"C:\Users\shind\Downloads\codes\codes\model.h5"

# Page config
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    h1 {
        color: #4B7BEC;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .stButton button {
        background-color: #4B7BEC;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #3867D6;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model or use dummy
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.warning(f"⚠ Model file not found at: {MODEL_PATH}. Using dummy predictions for now.")
    model = None

# Class labels
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Title
st.title("🩺 Diabetic Retinopathy Detection")
st.write("<h4 style='text-align:center;'>Upload a retinal fundus image to detect DR severity</h4>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Choose a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if model is not None:
        prediction = model.predict(img_array)
    else:
        # Dummy prediction (random values)
        prediction = np.random.dirichlet(np.ones(len(CLASS_NAMES)), size=1)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    with col2:
        st.markdown(f"### 🩻 **Prediction:** {predicted_class}")
        st.markdown(f"### 📊 **Confidence:** {confidence:.2f}%")

        # Plot all class probabilities
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, prediction[0], color="#4B7BEC")
        ax.set_ylabel("Probability")
        ax.set_title("Confidence per Class")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    st.success("✅ Analysis complete! Scroll down to see detailed results.")
