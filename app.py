import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ✅ Load YOLOv8 model
model = YOLO("weights/best.pt")

# ✅ App settings
st.set_page_config(page_title="Fabric Defect Detection", layout="wide")
st.title("🧵 Fabric Defect Detection with YOLOv8")

# === 📌 Sidebar Info ===
with st.sidebar:
    st.image("banner.jpg", caption="Example: Defective Fabric", use_container_width=True)
    st.markdown("""
    ### Why Fabric Defect Detection?

    Fabric defects like holes, tears, oil stains, and misweaves can degrade product quality,
    increase waste, and harm brand reputation.

    This app uses a YOLOv8 model trained on the AITEX dataset to detect such defects.

    📤 Upload a fabric image  
    📷 Or use your webcam for **real-time detection**
    """)

# === 🧠 Detection Function ===
def detect_defects(image):
    results = model(image)[0]
    return results.plot()

# === 📤 Upload Mode ===
if st.radio("Choose Input Method:", ["Upload Image", "Use Webcam"]) == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)
        if st.button("Detect Defects"):
            output = detect_defects(np.array(image))
            st.image(output, caption="Detection Output", use_container_width=True)

# === 📷 Real-Time Webcam Detection Mode ===
else:
    st.markdown("### Live Webcam Feed (Real-time YOLOv8 Detection)")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)[0]
            annotated_frame = results.plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
    key="realtime-detection",
    mode=WebRtcMode.SENDRECV,  # ✅ Enum, not string
    rtc_configuration=rtc_config,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    )


# === Footer ===
st.markdown("---")
st.markdown("🔧 Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · 🧠 Trained on AITEX Fabric Defect Dataset")
