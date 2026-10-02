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

def predict(img):
    img_resized = img.resize((128, 128))
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    label = "Malicious" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

st.title("QR Phishing Detector")
st.write("Ek ya ek se zyada QR code images upload karo, model batayega ki wo benign hai ya malicious.")

uploaded_files = st.file_uploader(
    "QR image(s) chuno",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) == 1:
        img = Image.open(uploaded_files[0]).convert("RGB")
        st.image(img, caption="Uploaded QR", width=250)
        label, confidence = predict(img)
        st.subheader(f"Result: {label}")
        st.write(f"Confidence: {confidence*100:.1f}%")
    else:
        results = []
        cols = st.columns(4)
        for i, f in enumerate(uploaded_files):
            img = Image.open(f).convert("RGB")
            label, confidence = predict(img)
            results.append({
                "File": f.name,
                "Result": label,
                "Confidence": f"{confidence*100:.1f}%"
            })
            with cols[i % 4]:
                st.image(img, caption=f.name, width=120)
                st.write(f"**{label}** ({confidence*100:.1f}%)")

        st.subheader("Summary")
        st.table(results)
