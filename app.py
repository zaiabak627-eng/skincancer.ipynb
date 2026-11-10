import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
from PIL import Image
import numpy as np

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="SkinGuard AI - Skin Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    .metric-card {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Robust Model Loading via gdown (Bypasses Google Drive Warning)
@st.cache_resource
def load_our_model():
    output = 'skin_cancer_model.keras'
    file_id = '1CkoJdzLhCgMjHt3mOjAr-vtKZ6dNiBdq'
    
    # اگر ماڈل فائل موجود نہیں ہے یا نامکمل ہے تو gdown کے ذریعے ڈاؤن لوڈ کرے گا
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        if os.path.exists(output):
            os.remove(output)
            
        url = f'https://drive.google.com/uc?id={file_id}'
        # fuzzy=True گوگل ڈرائیو کے وائرس/سیکیورٹی وارننگ پیجز کو ہینڈل کر لیਦਾ ہے
        gdown.download(url, output, quiet=False, fuzzy=True)
        
    model = load_model(output)
    return model

# Safely load the model
model_loaded = False
model = None
try:
    with st.spinner("Downloading and loading AI Model... Please wait..."):
        model = load_our_model()
        model_loaded = True
except Exception as e:
    model_loaded = False

# 3. Sidebar Navigation & Design
st.sidebar.title("🩺 SkinGuard AI")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("📂 Navigation Menu", ["Dashboard", "Image Analysis", "Skin Conditions"])

st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Information")
st.sidebar.text("Architecture: Convolutional Neural Network")
st.sidebar.text("Framework: TensorFlow / Keras")

if model_loaded:
    st.sidebar.markdown("**Model Status:** <span style='color:#00ffcc; font-weight:bold;'>● Online & Ready</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**Model Status:** <span style='color:#ff4b4b; font-weight:bold;'>● Offline (Error)</span>", unsafe_allow_html=True)

# 4. Main App Pages with Enhanced UI
if app_mode == "Dashboard":
    st.title("🩺 Welcome to SkinGuard AI")
    st.markdown("### Advanced Dermatological Lesion Classification System")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🚀 System Status</h3>
        """, unsafe_allow_html=True)
        if model_loaded:
            st.success("All pipelines are fully operational. The CNN model is successfully loaded and ready to evaluate skin lesion images.")
        else:
            st.error("Model Loading Error: Could not fetch the model from Google Drive. Please check file permissions.")
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>💡 How to Use</h3>
                <ol>
                    <li>Go to <b>Image Analysis</b> from the sidebar.</li>
                    <li>Upload a clear skin lesion image (.jpg, .jpeg, .png).</li>
                    <li>Click <b>Classify Image</b> to get AI-powered evaluation results.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

elif app_mode == "Image Analysis":
    st.title("🔍 Skin Lesion Image Analysis")
    st.markdown("Upload a skin image below to evaluate it using our trained deep learning model.")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_res = st.columns(2)
        
        with col_img:
            st.markdown("#### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col_res:
            st.markdown("#### 📊 Analysis Results")
            if st.button("🚀 Classify Image Now"):
                if model_loaded and model is not None:
                    with st.spinner("Analyzing dermatological features..."):
                        try:
                            # Preprocess image for CNN model
                            img = image.resize((224, 224))
                            img_array = np.array(img) / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                            
                            predictions = model.predict(img_array)
                            predicted_class = np.argmax(predictions, axis=1)
                            confidence = np.max(predictions) * 100
                            
                            st.success("Classification Successful!")
                            st.metric(label="Predicted Class Index", value=str(predicted_class[0]))
                            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.error("Model is offline or not loaded properly.")

elif app_mode == "Skin Conditions":
    st.title("📖 Skin Conditions Reference Guide")
    st.markdown("Explore detailed information regarding various skin lesions and CNN model diagnostics.")
    st.markdown("---")
    st.info("This application utilizes deep learning techniques to assist in preliminary skin lesion assessment. Always consult a certified dermatologist for professional medical diagnoses.")
