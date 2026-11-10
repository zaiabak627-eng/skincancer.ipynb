import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


st.set_page_config(
    page_title="SkinGuard AI | Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_cancer_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# Set this to match your Kaggle CNN preprocessing.
# True if training images were divided by 255; False if the model
# already contains a Rescaling layer.
NORMALIZE_TO_01 = True


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 10% 5%, #153c3a 0%, transparent 28%),
        linear-gradient(135deg, #071512 0%, #0b1f1b 55%, #102b25 100%);
    color: #f1f5f2;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102b25, #071512);
    border-right: 1px solid #25483d;
}
.hero {
    padding: 35px;
    border: 1px solid #315b4c;
    border-radius: 24px;
    background: linear-gradient(120deg, #173d32, #10251f);
    box-shadow: 0 15px 45px rgba(0,0,0,.25);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: #f5f5e9;
    margin-bottom: 8px;
}
.hero p {
    color: #b8d0c3;
    font-size: 16px;
}
.metric-card {
    background: linear-gradient(145deg, #17382e, #10251f);
    border: 1px solid #315b4c;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 12px;
}
.metric-label {
    color: #a9c6b6;
    font-size: 13px;
}
.metric-value {
    color: #e8d8a8;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}
.section-title {
    color: #e8d8a8;
    font-size: 23px;
    font-weight: 700;
    margin: 22px 0 12px 0;
}
.stButton > button {
    background: linear-gradient(90deg, #d6b878, #f0dba4);
    color: #10251f;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 800;
    width: 100%;
}
.stButton > button:hover {
    background: #f5e5b9;
    color: #071512;
    transform: translateY(-2px);
}
div[data-testid="stFileUploader"] {
    background: #132d25;
    border: 1px dashed #6e9c80;
    border-radius: 15px;
    padding: 15px;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


DISEASE_INFO = {
    "mel": {
        "name": "Melanoma",
        "description": "A potentially serious form of skin cancer.",
        "signs": "A changing mole, irregular borders, uneven colour or changing size."
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "description": "Commonly known as moles; usually benign.",
        "signs": "Often appear as brown, black or skin-coloured spots."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "A common type of skin cancer.",
        "signs": "A shiny bump, pink growth or sore that does not heal."
    },
    "akiec": {
        "name": "Actinic Keratoses",
        "description": "Sun-damage-related skin lesions that may require medical assessment.",
        "signs": "Rough, scaly or crusty patches on sun-exposed skin."
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "A group of generally benign keratotic skin lesions.",
        "signs": "May look like waxy, rough or slightly raised spots."
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "A usually benign skin growth.",
        "signs": "Often appears as a firm, small, brownish or reddish bump."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "A lesion involving blood vessels.",
        "signs": "May appear red, purple or bluish."
    }
}


@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty list.")

    return names


def prepare_image(image, model):
    image = ImageOps.exif_transpose(image).convert("RGB")
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 224
    width = input_shape[2] or 224
    image = image.resize((width, height))

    image_array = np.asarray(image, dtype=np.float32)

    if NORMALIZE_TO_01:
        image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


with st.sidebar:
    st.markdown("## 🔬 SkinGuard AI")
    st.caption("DEEP LEARNING • IMAGE CLASSIFICATION")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 🏠 Dashboard
    - 🖼️ Image Analysis
    - 📊 Prediction Results
    - 📚 Skin Conditions
    """)
    st.markdown("---")
    st.markdown("### Model Information")
    st.write("Model: CNN")
    st.write("Task: Skin lesion classification")
    st.write("Dataset: Configure according to your training dataset")
    st.markdown("---")
    st.caption("Educational research project")


st.markdown("""
<div class="hero">
    <h1>SkinGuard AI</h1>
    <p>AI-powered skin lesion image classification using a
    Convolutional Neural Network.</p>
    <p>Upload an image to explore the model's predicted class.</p>
</div>
""", unsafe_allow_html=True)


if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.info(
        "Upload skin_cancer_model.keras to the same GitHub "
        "repository folder as app.py."
    )
    st.stop()

if not CLASS_NAMES_PATH.exists():
    st.error("class_names.json file not found.")
    st.info(
        "Upload class_names.json containing the class names "
        "in exactly the same order used during model training."
    )
    st.stop()

try:
    model = load_trained_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load model or class names: {error}")
    st.stop()


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL TYPE</div>
        <div class="metric-value">CNN</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">OUTPUT CLASSES</div>
        <div class="metric-value">{len(class_names)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ANALYSIS MODE</div>
        <div class="metric-value">Image AI</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">🖼️ Upload Skin Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear JPG or PNG image."
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Uploaded Image")
            st.image(
                original_image,
                caption="Your uploaded image",
                use_container_width=True
            )

        with right:
            st.markdown("### Image Details")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Image format:** {original_image.format}")
            st.write(
                f"**Image dimensions:** "
                f"{original_image.size[0]} × {original_image.size[1]}"
            )
            st.markdown("---")
            analyze = st.button("🔍 Analyze Image")

        if analyze:
            with st.spinner("Analyzing image with CNN..."):
                _, image_array = prepare_image(original_image, model)
                predictions = model.predict(image_array, verbose=0)
                scores = np.asarray(predictions).reshape(-1)

                if len(scores) != len(class_names):
                    st.error(
                        "Model output count does not match class_names.json. "
                        "Check your training class order."
                    )
                    st.stop()

                # Convert logits to probabilities when model output is not
                # already a probability distribution.
                if (
                    np.any(scores < 0)
                    or np.any(scores > 1)
                    or not np.isclose(np.sum(scores), 1.0, atol=0.05)
                ):
                    scores = tf.nn.softmax(scores).numpy()
                else:
                    scores = scores / np.sum(scores)

                best_index = int(np.argmax(scores))
                predicted_code = str(class_names[best_index]).lower()

                disease = DISEASE_INFO.get(
                    predicted_code,
                    {
                        "name": str(class_names[best_index]),
                        "description": "No additional description is available.",
                        "signs": "Please consult a qualified dermatologist."
                    }
                )
                confidence = float(scores[best_index]) * 100

            st.markdown("---")
            st.markdown(
                '<div class="section-title">📊 Prediction Results</div>',
                unsafe_allow_html=True
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown("### Predicted Class")
                st.success(disease["name"])
                st.metric("Model confidence", f"{confidence:.2f}%")

            with result_col2:
                st.markdown("### About This Class")
                st.write(disease["description"])
                st.write("**Common visual signs:**")
                st.write(disease["signs"])

            st.markdown("### Class Probability Scores")
            chart_data = pd.DataFrame({
                "Class": [
                    DISEASE_INFO.get(str(name).lower(), {}).get(
                        "name", str(name)
                    )
                    for name in class_names
                ],
                "Probability (%)": scores * 100
            }).sort_values("Probability (%)", ascending=False)

            st.bar_chart(
                chart_data.set_index("Class"),
                horizontal=True
            )

            st.download_button(
                "⬇️ Download Prediction Results",
                data=chart_data.to_csv(index=False).encode("utf-8"),
                file_name="skin_cancer_prediction.csv",
                mime="text/csv"
            )

            st.warning(
                "This is an educational AI prediction, not a medical "
                "diagnosis. Model confidence is not the probability that "
                "a person has cancer. Consult a qualified dermatologist "
                "for medical assessment."
            )

    except Exception as error:
        st.error(f"Image analysis failed: {error}")
else:
    st.info("Upload a JPG or PNG image to begin analysis.")


st.markdown("---")
st.markdown(
    '<div class="section-title">📚 Skin Condition Information</div>',
    unsafe_allow_html=True
)

for code, info in DISEASE_INFO.items():
    with st.expander(info["name"]):
        st.write(info["description"])
        st.write("**Common visual signs:**", info["signs"])


st.markdown("---")
st.caption(
    "SkinGuard AI | CNN Skin Lesion Classification | "
    "Academic and educational use only"
)
