import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
import numpy as np
import joblib
import nest_asyncio
import asyncio

# Apply nest_asyncio to fix the event loop issue
nest_asyncio.apply()

# Load the YOLOv8 model trained on emotions (best.pt should be a model trained for emotion detection)
model = YOLO('emotionaldetect.pt')

# Function to detect emotions in video
def detect_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 model
        results = model.predict(frame)
        
        # Process results
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class index
                label = model.names[cls]  # Get label (emotions like "happy", "sad", etc.)
                
                # Ensure it's one of the emotions you want to detect (happy, sad, neutral)
                if label in ['Angry', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']:  # Adjust the labels based on your trained model
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        stframe.image(frame, channels="RGB")
        
    cap.release()

# Streamlit UI
st.title("Emotion Detection in Video using YOLOv8")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    detect_emotions(tfile.name)
