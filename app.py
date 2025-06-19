import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
import av
import cv2  # ✅ required for resizing

# ✅ Load YOLOv8 model
model = YOLO("weights/best.pt")

# ✅ Streamlit page config
st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_custom_theme():
    st.markdown("""
        <style>
            body {
                background-color: #0f1117;
                color: #e0e0e0;
            }
            [data-testid="stSidebar"] {
                background-color: #161a24;
            }
            h1, h2, h3 {
                color: #00f7ff;
            }
            .markdown-text-container {
                color: #e0e0e0;
            }
            .stButton > button {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5em 1em;
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #00bfff;
                color: black;
            }
            img {
                border-radius: 10px;
            }
            .stRadio > div {
                background-color: #1a1d2b;
                padding: 0.5em;
                border-radius: 10px;
            }
            h1 {
                margin-bottom: 0.1em !important;
            }
        </style>
    """, unsafe_allow_html=True)

set_custom_theme()

# === Title
st.title("Fabric Defect Detection with YOLOv8")
st.markdown("The Fabric Defect Detector is a real-time AI-based web app built using YOLOv8 to identify fabric defects like holes, tears, and stains from images or live webcam input. Trained on the AITEX dataset and developed using Streamlit, it offers a fast and user-friendly solution for textile quality inspection.")

# === 📌 Sidebar Info ===
with st.sidebar:
    st.image("banner.jpg", caption="Example: Defective Fabric", use_container_width=True)
    st.markdown("""
## Team Details
1) Kirtan Mudaliyar  
2) Namrata Rathod  
3) Anshal Suthar  
4) Akansha Ravat  
5) Dishant Modi
    """)

# === 🧠 Detection Function (Single Image)
def detect_defects(image):
    results = model(image)[0]
    return results.plot()

# === 🔘 Input Method
input_mode = st.radio("Choose Input Method:", ["Upload Image", "Real-Time tracking"])

# === 📤 Upload Image
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width=400)
            detect = st.button("Detect Defects")

        with col2:
            if 'detect' in locals() and detect:
                output = detect_defects(np.array(image))
                st.image(output, caption="Detection Output", width=400)

# === 📷 Real-Time Webcam Mode
else:
    st.markdown("### Live Webcam Feed (YOLOv8 Real-Time Detection)")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # ✅ Resize for performance boost
            img = cv2.resize(img, (640, 480))

            # ✅ Predict with lower resolution and faster settings
            results = model.predict(img, imgsz=640, conf=0.4, verbose=False)[0]
            annotated = results.plot()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="realtime-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    )

# === Footer
st.markdown("---")
st.markdown("🔧 P
