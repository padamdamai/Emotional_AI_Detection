import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
import numpy as np
import time

# Load the YOLOv8 model trained on emotions (best.pt should be a model trained for emotion detection)
model = YOLO('emotionaldetect.pt')

# Function to detect emotions in video
def detect_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    # Video reading and processing loop
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every N-th frame to improve performance (change this value to skip more frames)
        if frame_counter % 10 == 0:
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

            # Convert to RGB for displaying in Streamlit (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            stframe.image(frame, channels="RGB")

            # Add a small delay to slow down the video processing
            time.sleep(0.1)  # Adjust this delay value as needed

        frame_counter += 1

    cap.release()

# Streamlit UI
st.title("Emotion Detection in Video using YOLOv8")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    st.write("Processing video...")
    detect_emotions(tfile.name)
