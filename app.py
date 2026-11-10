import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# یہ کوڈ خود بخود موجودہ فولڈر کا صحیح پاتھ تلاش کر لے گا
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.keras")

# اب ماڈل کو اس درست پاتھ سے لوڈ کریں
model = load_model(MODEL_PATH)
