import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SkinGuard AI",
    page_icon="🩺",
    layout="wide"
)

# Dynamically resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.keras")

@st.cache_resource
def load_lesion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return load_model(MODEL_PATH)

# Load model with error handling
try:
    model = load_lesion_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Sidebar Navigation
st.sidebar.title("SkinGuard AI")
st.sidebar.markdown("DEEP LEARNING • IMAGE CLASSIFICATION")
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Image Analysis", "Skin Conditions"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.markdown("**Model:** CNN")
st.sidebar.markdown("**Task:** Skin lesion classification")

if page == "Dashboard":
    st.markdown("""
        <div style='background-color: #1b3022; padding: 25px; border-radius: 10px;'>
            <h1 style='color: white;'>SkinGuard AI</h1>
            <p style='color: #d1d5db;'>AI-powered skin lesion image classification using a Convolutional Neural Network.</p>
            <p style='color: #d1d5db;'>Upload an image to explore the model's predicted class.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error(f"Could not load model or class names: {load_error}")
    else:
        st.success("Model loaded successfully and ready for predictions!")

elif page == "Image Analysis":
    st.header("Skin Lesion Image Analysis")
    
    if not model_loaded:
        st.error(f"Could not load model: {load_error}")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Image"):
                with st.spinner("Analyzing image..."):
                    # Preprocessing image for CNN model (adjust target size according to your training setup, e.g., 224x224)
                    img_resized = image.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    confidence = float(np.max(predictions[0])) * 100
                    
                    st.success(f"Prediction Complete!")
                    st.write(f"**Predicted Class Index:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2f}%")

elif page == "Skin Conditions":
    st.header("Skin Conditions Reference Guide")
    st.markdown("""
    - **Melanoma (mel):** A potentially serious skin cancer requiring medical assessment.
    - **Melanocytic Nevi (nv):** Usually stable, round or oval pigmented moles.
    - **Basal Cell Carcinoma (bcc):** A common type of skin cancer.
    - **Actinic Keratoses (akiec):** Rough, scaly or crusty patches.
    - **Benign Keratosis (bkl):** Waxy, rough or stuck-on appearing growth.
    - **Dermatofibroma (df):** Small, firm skin-coloured or brown bump.
    - **Vascular Lesions (vasc):** Red, purple or bluish skin marks.
    """)
