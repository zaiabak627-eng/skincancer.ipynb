import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SkinGuard AI | Deep Learning",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for modern styling and nice dark-green theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1713;
    }
    .stSidebar {
        background-color: #122019;
    }
    .hero-card {
        background: linear-gradient(135deg, #1b3022 0%, #112217 100%);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #2d4a37;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        color: #a3b8ad;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .custom-box {
        background-color: #16261d;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #264231;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2e7d54 0%, #1e5337 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(46, 125, 84, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #358f60 0%, #246342 100%);
        box-shadow: 0 6px 16px rgba(46, 125, 84, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

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
st.sidebar.markdown("<h2 style='color: #4ade80;'>SkinGuard AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>DEEP LEARNING • IMAGE CLASSIFICATION</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Image Analysis", "Skin Conditions"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info("**Model:** CNN\n\n**Task:** Skin lesion classification\n\n**Status:** " + ("🟢 Online" if model_loaded else "🔴 Offline"))

if page == "Dashboard":
    st.markdown("""
        <div class="hero-card">
            <div class="hero-title">SkinGuard AI 🩺</div>
            <div class="hero-subtitle">
                AI-powered skin lesion image classification using an advanced Convolutional Neural Network.<br>
                Upload an image in the <b>Image Analysis</b> section to explore the model's predicted class and confidence score.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="custom-box">
                <h3 style='color: #4ade80; margin-top:0;'>🚀 System Status</h3>
                <p style='color: #d1d5db;'>Model loading and processing pipelines are fully operational and ready to evaluate dermatological images.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="custom-box">
                <h3 style='color: #4ade80; margin-top:0;'>💡 How to Use</h3>
                <p style='color: #d1d5db;'>1. Navigate to <b>Image Analysis</b>.<br>2. Upload a clear skin lesion image (.jpg, .jpeg, .png).<br>3. Click <b>Classify Image</b> to view results.</p>
            </div>
        """, unsafe_allow_html=True)
        
    if not model_loaded:
        st.error(f"⚠️ Model Loading Error: {load_error}")

elif page == "Image Analysis":
    st.markdown("<h2 style='color: #ffffff;'>🔍 Skin Lesion Image Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af;'>Upload a skin lesion image below to run deep learning classification.</p>", unsafe_allow_html=True)
    
    if not model_loaded:
        st.error(f"Could not load model: {load_error}")
    else:
        uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col_img, col_res = st.columns(2)
            
            with col_img:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Sample", use_column_width=True)
            
            with col_res:
                st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
                st.markdown("### Analysis Panel")
                if st.button("Classify Image"):
                    with st.spinner("Processing through Convolutional Neural Network..."):
                        img_resized = image.resize((224, 224))
                        import os
import os
import tensorflow as tf
from tensorflow.keras.models.saving import load_model
import requests
import streamlit as st

@st.cache_resource
def load_our_model():
    output = 'skin_cancer_model.keras'
    if not os.path.exists(output):
        # گوگل ڈرائیو سے براہ راست فائل ڈاؤن لوڈ کرنے کا محفوظ طریقہ
        file_id = '1CkoJdzLhCgMjHt3mOjAr-vtKZ6dNiBdq'
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # سیشن کے ذریعے بڑی فائل ڈاؤن لوڈ کرنا
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # اگر کوئی کنفرمیشن ٹوکن ہو تو اسے ہینڈل کریں
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
            
        # فائل کو ٹکڑوں میں محفوظ کریں
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    
    model = load_model(output)
    return model

model = load_our_model()
model = load_our_model()
