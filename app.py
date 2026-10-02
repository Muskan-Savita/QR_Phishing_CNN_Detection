import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os
import cv2

MODEL_PATH = "qr_cnn_clean_model.keras"
DRIVE_FILE_ID = "1d4wJzFv0QFtyQS9GtiRc9OLNFPNsSGiY"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    m = tf.keras.models.load_model(MODEL_PATH)
    # Warm-up call: forces Keras to fully build the model's internal graph.
    # Without this, reading an intermediate layer's output (for Grad-CAM)
    # can raise an AttributeError on freshly loaded models.
    _ = m.predict(np.zeros((1, 128, 128, 3), dtype="float32"), verbose=0)
    return m

model = load_model()

def get_last_conv_layer_name(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

LAST_CONV_LAYER = get_last_conv_layer_name(model)

def predict(img):
    img_resized = img.resize((128, 128))
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    label = "Malicious" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence, arr

def make_gradcam(img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(pil_img, heatmap):
    img = np.array(pil_img.convert("RGB").resize((128, 128)))
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return overlay

def show_single_result(img):
    label, confidence, arr = predict(img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original QR", width=250)
    with col2:
        if LAST_CONV_LAYER:
            try:
                heatmap = make_gradcam(arr, LAST_CONV_LAYER)
                overlay = overlay_heatmap(img, heatmap)
                st.image(overlay, caption="Model kahan dekh raha hai", width=250)
            except Exception as e:
                st.write("Heatmap is baar nahi ban paya.")
        else:
            st.write("Heatmap uplabdh nahi (conv layer nahi mila)")

    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {confidence*100:.1f}%")

st.title("QR Phishing Detector")
st.write("QR code image upload karo ya camera se scan karo, model batayega ki wo benign hai ya malicious.")

tab1, tab2 = st.tabs(["Upload", "Camera"])

with tab1:
    uploaded_files = st.file_uploader(
        "QR image(s) chuno",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            img = Image.open(uploaded_files[0]).convert("RGB")
            show_single_result(img)
        else:
            results = []
            cols = st.columns(4)
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                label, confidence, _ = predict(img)
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

with tab2:
    st.write("Camera se seedha QR scan karo.")
    cam_file = st.camera_input("QR code ko camera ke saamne rakho")
    if cam_file is not None:
        img = Image.open(cam_file).convert("RGB")
        show_single_result(img)
