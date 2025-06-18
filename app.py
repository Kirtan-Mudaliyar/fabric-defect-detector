import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# ✅ Load YOLOv8 model
model = YOLO("weights/best.pt")

# ✅ App settings
st.set_page_config(page_title="Fabric Defect Detection", layout="wide")
st.title("Fabric Defect Detection with YOLOv8")

# === 📌 Sidebar Info ===
with st.sidebar:
    st.image("banner.jpg", caption="Example: Defective Fabric", use_container_width=True)
    st.markdown("""
    ### Why Fabric Defect Detection?

    Fabric defects like holes, tears, oil stains, and misweaves can degrade product quality,
    increase waste, and harm brand reputation.

    This app uses a YOLOv8 model trained on the AITEX dataset to detect such defects.

    Upload a fabric image or use your webcam to test detection in real-time!
    """)

# === 🧠 Detection Function ===
def detect_defects(image):
    results = model(image)[0]
    return results.plot()

# === 🔘 Input Selector ===
input_mode = st.radio("Choose Input Method:", ["Upload Image", "Use Webcam"])

# === 📤 Upload Mode ===
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True,width=400)
        if st.button("Detect Defects"):
            output = detect_defects(np.array(image))
            st.image(output, caption="Detection Output", use_container_width=True)

# === 📷 Webcam Mode ===
elif input_mode == "Use Webcam":
    webcam_image = st.camera_input("Capture Fabric Sample")
    if webcam_image is not None:
        image = Image.open(webcam_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)
        if st.button("Detect Defects from Webcam"):
            output = detect_defects(np.array(image))
            st.image(output, caption="Detection Output", use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · Trained on AITEX Fabric Defect Dataset")
