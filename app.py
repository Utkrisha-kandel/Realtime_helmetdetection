import cv2
import streamlit as st
import time
from service.detection_pipeline import HelmetDetectionPipeline

st.set_page_config(page_title="Helmet Detection", layout="wide")
st.title("Helmet Detection")

pipeline = HelmetDetectionPipeline("models/best.pt", conf_thresh=0.25)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not detected.")
        break

    annotated, message = pipeline.detect(frame)

    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame_rgb.copy(), channels="RGB", use_container_width=True)

    time.sleep(0.03)

cap.release()
