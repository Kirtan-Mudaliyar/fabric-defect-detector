import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,  # ✅ REQUIRED for mode=WebRtcMode.SENDRECV
)
import av

# ✅ Load YOLOv8 model
model = YOLO("weights/best.pt")

# ✅ Streamlit page config
st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)
def set_custom_theme():
    st.markdown("""
        <style>
            body {
                background-color: #0d0d0d;
                color: #d0ffd0;
                font-family: 'Courier New', monospace;
            }

            [data-testid="stSidebar"] {
                background-color: #111;
            }

            h1, h2, h3 {
                color: #00ff41;
                text-shadow: 0 0 10px #00ff41;
            }

            .markdown-text-container {
                color: #d0ffd0;
            }

            .stButton > button {
                background-color: #00cc44;
                color: black;
                border: none;
                border-radius: 6px;
                padding: 0.5em 1em;
                font-weight: bold;
                transition: 0.3s ease;
            }

            .stButton > button:hover {
                background-color: #00ff88;
                color: black;
            }

            img {
                border-radius: 8px;
            }

            .stRadio > div {
                background-color: #1a1a1a;
                padding: 0.5em;
                border-radius: 8px;
            }

            .css-1offfwp {
                color: #00ff41 !important;
            }
        </style>
    """, unsafe_allow_html=True)

set_custom_theme()

st.title("Fabric Defect Detection with YOLOv8")

# === 📌 Sidebar Info ===
with st.sidebar:
    st.image("banner.jpg", caption="Example: Defective Fabric", use_container_width=True)
    st.markdown("""
## About the Fabric Defect Detector
The Fabric Defect Detector is a real-time AI-based web app built using YOLOv8 to identify fabric defects like holes, tears, and stains from images or live webcam input. Trained on the AITEX dataset and developed using Streamlit, it offers a fast and user-friendly solution for textile quality inspection.
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

# === Input Method Selection
input_mode = st.radio("Choose Input Method:", ["Upload Image", "Real-Time tracking"])

# === 📤 Upload Mode
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width=400)
            detect = st.button("Detect Defects")  # 👈 Button just below input image

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
            results = model(img)[0]
            annotated_frame = results.plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="realtime-detection",
        mode=WebRtcMode.SENDRECV,  # ✅ Enum instead of string
        rtc_configuration=rtc_config,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# === Footer
st.markdown("---")
st.markdown("Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · Trained on AITEX Fabric Defect Dataset")
