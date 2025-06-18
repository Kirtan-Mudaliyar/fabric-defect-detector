---
title: Fabric Defect Detector 🧵
emoji: 🧶
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.34.0
app_file: app.py
pinned: false
license: mit
---

# 🧵 Fabric Defect Detection with YOLOv8

This Streamlit app detects common fabric defects using a custom-trained YOLOv8 model.  
Users can either upload an image or use a webcam to analyze fabric samples.

---

## 🚀 Features
- 📷 Real-time webcam detection
- 📁 Upload image for inference
- ⚙️ Powered by Ultralytics YOLOv8
- 🧠 Trained on AITEX Fabric Defect Dataset

---

## 🛠 Usage

### 📦 On Hugging Face Spaces
Click the “Run” button on the top of this page and try:

- Uploading images of fabric with defects
- Live detection using webcam

### 🖥️ Run Locally

```bash
git clone https://huggingface.co/spaces/kirtan0706/fabric-defect-detector
cd fabric-defect-detector
pip install -r requirements.txt
python app.py