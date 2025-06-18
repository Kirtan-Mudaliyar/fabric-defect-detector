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
st.set_page_config(page_title="Fabric Defect Detection", layout="wide")
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
        st.image(image, caption="Uploaded Image",length=400,width=400)
        if st.button("Detect Defects"):
            output = detect_defects(np.array(image))
            st.image(output, caption="Detection Output", use_container_width=True)

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
