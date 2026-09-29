import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

MODEL_PATH = "qr_cnn_clean_model.keras"
DRIVE_FILE_ID = "1d4wJzFv0QFtyQS9GtiRc9OLNFPNsSGiY"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("QR Phishing Detector")
st.write("Ek QR code image upload karo, model batayega ki wo benign hai ya malicious.")

uploaded_file = st.file_uploader("QR image chuno", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded QR", width=250)

    img_resized = img.resize((128, 128))
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]
    label = "Malicious" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred

    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {confidence*100:.1f}%")
