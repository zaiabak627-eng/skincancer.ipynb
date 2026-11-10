import os
import json
import hashlib
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ----------------------------- APP CONFIG -----------------------------
st.set_page_config(
    page_title="SkinGuard AI | Skin Lesion Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "skin_cancer_model.keras")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / MODEL_FILENAME)))
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
DEFAULT_DRIVE_ID = "1CkoJdzLhCgMjHt3mOjAr-vtKZ6dNiBdq"
IMAGE_SIZE = (224, 224)

CLASS_INFO = {
    "akiec": ("Actinic keratoses / intraepithelial carcinoma", "A precancerous or early skin lesion category in the dataset."),
    "bcc": ("Basal cell carcinoma", "A common type of skin cancer."),
    "bkl": ("Benign keratosis-like lesions", "A group of usually non-cancerous lesion patterns."),
    "df": ("Dermatofibroma", "A usually benign skin growth."),
    "mel": ("Melanoma", "A serious form of skin cancer that needs professional assessment."),
    "nv": ("Melanocytic nevi", "Common mole-related patterns."),
    "vasc": ("Vascular lesions", "Lesions related to blood vessels."),
}

# ----------------------------- STYLING -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080f1c; --panel:#101b2b; --panel2:#142338; --line:#26384e;
        --text:#edf5ff; --muted:#9fb1c8; --teal:#36d6c3; --blue:#65a9ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% 0%, #142943 0%, #0a1423 42%, #080f1c 100%); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0c1727,#0a1220); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1450px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.03em; color:var(--text); }
.hero { padding:30px 34px; border:1px solid #29415c; border-radius:22px;
        background:linear-gradient(115deg,rgba(25,54,81,.96),rgba(15,31,50,.92));
        box-shadow:0 18px 55px rgba(0,0,0,.22); margin-bottom:24px; }
.eyebrow { color:var(--teal); font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Manrope',sans-serif; font-size:2.3rem; font-weight:800; line-height:1.15; margin:.45rem 0; }
.hero-sub { color:#b7c8dc; font-size:1.02rem; max-width:780px; }
.panel { background:linear-gradient(145deg,rgba(20,35,56,.96),rgba(14,26,43,.96));
         border:1px solid var(--line); border-radius:18px; padding:22px 24px;
         box-shadow:0 12px 32px rgba(0,0,0,.16); }
.panel-title { font-family:'Manrope',sans-serif; font-size:1.1rem; font-weight:800; margin-bottom:10px; }
.muted { color:var(--muted); }
.status-pill { display:inline-block; padding:7px 12px; border-radius:99px; font-size:.8rem; font-weight:700; }
.good { color:#6af0d5; background:rgba(54,214,195,.12); border:1px solid rgba(54,214,195,.35); }
.bad { color:#ff9a9a; background:rgba(255,100,100,.12); border:1px solid rgba(255,100,100,.3); }
div.stButton > button { width:100%; border:0; border-radius:12px; padding:.78rem 1rem;
    color:#06131e; font-weight:800; background:linear-gradient(90deg,#36d6c3,#69b6ff); }
div.stButton > button:hover { color:#06131e; filter:brightness(1.08); border:0; }
div[data-testid="stFileUploader"] { background:rgba(16,27,43,.75); border:1px dashed #47627e; border-radius:14px; padding:12px; }
div[data-testid="stMetric"] { background:#122239; border:1px solid #2a405a; padding:14px 16px; border-radius:14px; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def read_class_names():
    if CLASS_NAMES_PATH.exists():
        try:
            data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(x) for x in data]
        except Exception:
            pass
    return ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


CLASS_NAMES = read_class_names()


def get_secret(name, default=""):
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)


def download_model():
    """Load a local model first, otherwise fetch from Hugging Face or Drive."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return str(MODEL_PATH), "Local model file"

    # Recommended for deployment: store the large model in a Hugging Face model repo.
    hf_repo_id = get_secret("HF_REPO_ID", os.getenv("HF_REPO_ID", "")).strip()
    hf_filename = get_secret("HF_MODEL_FILENAME", MODEL_FILENAME).strip()
    hf_token = get_secret("HF_TOKEN", os.getenv("HF_TOKEN", "")).strip()

    if hf_repo_id:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type="model",
            token=hf_token or None,
        )
        return downloaded, f"Hugging Face Hub: {hf_repo_id}"

    # Drive fallback. The file must be shared as "Anyone with the link – Viewer".
    drive_id = get_secret("GDRIVE_FILE_ID", DEFAULT_DRIVE_ID).strip()
    if drive_id:
        import gdown
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(
            id=drive_id,
            output=str(MODEL_PATH),
            quiet=False,
            fuzzy=True,
            resume=True,
        )
        if not result or not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
            raise RuntimeError(
                "Google Drive did not return a valid model file. Check that the file is shared "
                "as 'Anyone with the link – Viewer', and verify the file ID."
            )
        return str(MODEL_PATH), "Google Drive"

    raise RuntimeError(
        "No model source configured. Add a local model, or set HF_REPO_ID in Streamlit Secrets."
    )


@st.cache_resource(show_spinner=False)
def load_ai_model():
    model_file, source = download_model()
    # compile=False avoids requiring the original training-time optimizer/loss config.
    loaded = tf.keras.models.load_model(model_file, compile=False)
    return loaded, source, os.path.getsize(model_file)


def model_status():
    try:
        model, source, size = load_ai_model()
        return model, source, size, None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def preprocess_image(pil_image):
    # RGB conversion prevents grayscale/RGBA uploads from producing wrong tensor shapes.
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), image


def pretty_class(code):
    return CLASS_INFO.get(code, (code.upper(), "Dataset class label."))[0]


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("## 🩺 SkinGuard AI")
    st.caption("DERMATOLOGY • AI RESEARCH ASSISTANT")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Image Analysis", "Class Reference"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("### System monitor")
    with st.spinner("Checking model..."):
        model, model_source, model_bytes, model_error = model_status()

    if model is not None:
        st.markdown('<span class="status-pill good">● MODEL ONLINE</span>', unsafe_allow_html=True)
        st.caption(f"Source: {model_source}")
        st.caption(f"Model size: {model_bytes / (1024 * 1024):.1f} MB")
    else:
        st.markdown('<span class="status-pill bad">● MODEL OFFLINE</span>', unsafe_allow_html=True)
        st.caption("The interface is available, but prediction is disabled until the model loads.")
    st.divider()
    st.caption("SkinGuard AI • Research prototype")


# ----------------------------- OVERVIEW -----------------------------
if page == "Overview":
    st.markdown("""
    <section class="hero">
      <div class="eyebrow">INTELLIGENT SKIN LESION ANALYSIS</div>
      <div class="hero-title">A clearer view of skin health.</div>
      <div class="hero-sub">SkinGuard AI is a deep-learning research interface for exploring
      image classification across seven skin-lesion categories.</div>
    </section>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">🧠 Model architecture</div><div class="muted">Convolutional Neural Network<br>TensorFlow / Keras</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="panel-title">🧬 Classification set</div><div class="muted">7 lesion categories<br>HAM10000 label mapping</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><div class="panel-title">🖼️ Image input</div><div class="muted">JPG, JPEG, PNG<br>224 × 224 model input</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">🚀 System status</div>', unsafe_allow_html=True)
        if model is not None:
            st.success(f"Model loaded successfully from {model_source}.")
        else:
            st.error("The model could not be loaded.")
            st.code(model_error or "Unknown model loading error")
            st.caption("Configure a Hugging Face model repository in Streamlit Secrets, or check the Google Drive sharing permission.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">✨ Quick start</div>
          <ol class="muted">
            <li>Open <b>Image Analysis</b>.</li>
            <li>Upload a clear JPG or PNG image.</li>
            <li>Review the model's class probabilities.</li>
            <li>Use the result only as a research aid.</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------- IMAGE ANALYSIS -----------------------------
elif page == "Image Analysis":
    st.markdown('<div class="eyebrow">ANALYSIS WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Skin lesion image analysis")
    st.markdown('<p class="muted">Upload an image to view the model output and class probability distribution.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], help="Use a clear, well-lit image.")
    if uploaded:
        try:
            pil_image = Image.open(uploaded)
            tensor, display_image = preprocess_image(pil_image)
            image_col, result_col = st.columns([1, 1.15], gap="large")
            with image_col:
                st.markdown('<div class="panel-title">📷 Uploaded image</div>', unsafe_allow_html=True)
                st.image(display_image, use_container_width=True)
                st.caption(f"Image size: {pil_image.width} × {pil_image.height}")
            with result_col:
                st.markdown('<div class="panel-title">📊 Model evaluation</div>', unsafe_allow_html=True)
                if st.button("Run image classification", type="primary", disabled=(model is None)):
                    with st.spinner("Evaluating image..."):
                        try:
                            raw = np.asarray(model.predict(tensor, verbose=0))
                            scores = raw[0].astype(float).reshape(-1)
                            if len(scores) != len(CLASS_NAMES):
                                raise ValueError(
                                    f"Model returned {len(scores)} outputs but class_names.json contains "
                                    f"{len(CLASS_NAMES)} labels. Confirm the training label order."
                                )
                            # Convert logits to probabilities only when outputs do not already sum to ~1.
                            if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=0.03):
                                exp = np.exp(scores - np.max(scores))
                                scores = exp / exp.sum()
                            else:
                                scores = scores / scores.sum()

                            ranking = np.argsort(scores)[::-1]
                            best_idx = int(ranking[0])
                            best_code = CLASS_NAMES[best_idx]
                            st.session_state["last_result"] = {
                                "code": best_code,
                                "score": float(scores[best_idx]),
                                "scores": scores.tolist(),
                            }
                        except Exception as exc:
                            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

                result = st.session_state.get("last_result")
                if result:
                    best_code = result["code"]
                    st.success("Model inference completed.")
                    st.metric("Top predicted class", pretty_class(best_code))
                    st.metric("Model score", f'{result["score"] * 100:.1f}%')
                    st.markdown("**Class probability distribution**")
                    chart_data = {
                        "Class": [f"{CLASS_NAMES[i]} — {pretty_class(CLASS_NAMES[i])}" for i in range(len(CLASS_NAMES))],
                        "Score": result["scores"],
                    }
                    st.bar_chart(chart_data, x="Class", y="Score", horizontal=True)
                    st.info("A model score is not a medical diagnosis or a measure of cancer risk. Please consult a qualified dermatologist for any concerning or changing skin lesion.")
                elif model is None:
                    st.warning("Prediction is unavailable because the AI model is offline. See the model status and error details in the sidebar/Overview.")
                else:
                    st.caption("Select **Run image classification** to evaluate this image.")
        except Exception as exc:
            st.error(f"Could not read this image: {type(exc).__name__}: {exc}")
    else:
        st.markdown('<div class="panel"><div class="panel-title">🔒 Your image stays in this session</div><div class="muted">Choose a JPG, JPEG, or PNG file to begin. Do not upload images containing identifying personal information.</div></div>', unsafe_allow_html=True)

# ----------------------------- CLASS REFERENCE -----------------------------
else:
    st.markdown('<div class="eyebrow">DATASET LABEL GUIDE</div>', unsafe_allow_html=True)
    st.title("Skin condition reference")
    st.markdown('<p class="muted">The following labels describe the seven output categories used by the model.</p>', unsafe_allow_html=True)
    for code in CLASS_NAMES:
        title, description = CLASS_INFO.get(code, (code.upper(), "Dataset class label."))
        with st.expander(f"{code.upper()}  ·  {title}"):
            st.write(description)
    st.warning("This project is an educational/research prototype. It must not be used to diagnose, rule out, or treat any skin condition. A qualified clinician must assess medical concerns.")

# ----------------------------- FOOTER -----------------------------
st.divider()
st.caption("SkinGuard AI • Educational research prototype • Not a medical device")
