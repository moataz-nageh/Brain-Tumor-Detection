import streamlit as st

st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

import cv2
import numpy as np
import joblib
from PIL import Image
import base64

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


bg_path = "background.png"  
bg_base64 = get_base64(bg_path)



st.markdown(f"""
<style>

.stApp {{
    background-image: 
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.35)),
        url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
}}

/* Header Box */
.header-box {{
    background: rgba(0, 0, 0, 0.6);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}}

.header-title {{
    font-size: 38px;
    font-weight: bold;
    color: white;
}}

.header-sub {{
    font-size: 18px;
    color: #ddd;
}}

/* Result Box */
.result-box {{
    background: rgba(255,255,255,0.9);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: black;
    backdrop-filter: blur(5px);
    margin-top: 20px;
}}

/* Upload + Select box styling */
.stFileUploader, .stSelectbox {{
    background: rgba(0,0,0,0.5);
    padding: 10px;
    border-radius: 10px;
    backdrop-filter: blur(5px);
}}

</style>
""", unsafe_allow_html=True)


# ===============================
# 🧠 Build CNN
# ===============================
def build_cnn():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model


# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.pkl")
    gb = joblib.load("gb_model.pkl")
    
    cnn = build_cnn()
    cnn.load_weights("cnn.weights.h5")
    
    return rf, gb, cnn


rf_model, gb_model, model = load_models()

class_names = ["glioma", "meningioma", "pituitary", "notumor"]


# ===============================
# UI
# ===============================

# Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🧠 Brain Tumor Detection</div>
    <div class="header-sub">Upload MRI image and choose model</div>
</div>
""", unsafe_allow_html=True)


# Upload
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

# Model selection
model_choice = st.selectbox(
    "🧠 Select Model",
    ["Random Forest", "Gradient Boosting"]
)


# ===============================
# Prediction
# ===============================
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Predict"):
        
        img_resized = cv2.resize(img, (128, 128)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        features = model.predict(img_resized)
        
        if model_choice == "Random Forest":
            prediction = rf_model.predict(features)
        else:
            prediction = gb_model.predict(features)
        
        predicted_class = class_names[prediction[0]]

        
        st.markdown(f"""
        <div class="result-box">
            Prediction: {predicted_class}
        </div>
        """, unsafe_allow_html=True)