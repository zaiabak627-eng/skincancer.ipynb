import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="SkinGuard AI",
    page_icon="🩺",
    layout="wide"
)

# 2. Robust Model Loading Function (Google Drive Direct Download via Requests)
@st.cache_resource
def load_our_model():
    output = 'skin_cancer_model.keras'
    file_id = '1CkoJdzLhCgMjHt3mOjAr-vtKZ6dNiBdq'
    
    # اگر فائل موجود نہ ہو یا کرپٹ ہو تو گوگل ڈرائیو سے ڈاؤن لوڈ کرے گا
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        if os.path.exists(output):
            os.remove(output)
            
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        session = requests.Session()
        response = session.get(url, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
            
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    
    model = load_model(output)
    return model

# Safely load the model
try:
    model = load_our_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", ["Dashboard", "Image Analysis", "Skin Conditions"])

# Sidebar Model Status
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.text("Model: CNN")
st.sidebar.text("Task: Skin lesion classification")
if model_loaded:
    st.sidebar.markdown("**Status:** <span style='color:green;'>● Online</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**Status:** <span style='color:red;'>● Offline</span>", unsafe_allow_html=True)

# 4. Main App Pages
if app_mode == "Dashboard":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚀 System Status")
        if model_loaded:
            st.success("Model loading and processing pipelines are fully operational and ready to evaluate dermatological images.")
        else:
            st.error("Model Loading Error: Model file could not be loaded from Google Drive.")
            
    with col2:
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. Navigate to **Image Analysis**.
        2. Upload a clear skin lesion image (.jpg, .jpeg, .png).
        3. Click **Classify Image** to view results.
        """)

elif app_mode == "Image Analysis":
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload a clear skin lesion image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify Image"):
            if model_loaded:
                with st.spinner("Analyzing image..."):
                    try:
                        # Image preprocessing for CNN model
                        img = image.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions, axis=1)
                        
                        st.success(f"Classification complete! Prediction result class index: {predicted_class[0]}")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
            else:
                st.error("Model is not loaded, cannot perform classification.")

elif app_mode == "Skin Conditions":
    st.header("Skin Conditions Information")
    st.write("Here you can find educational information regarding various skin conditions, lesion types, and convolutional neural network evaluation details.")
