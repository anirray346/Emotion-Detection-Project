import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Emotion Detector AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a colorful and modern look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stHeader {
        color: #2e86c1;
    }
    .emotion-label {
        font-size: 24px;
        font-weight: bold;
        color: #8e44ad;
        text-align: center;
        padding: 10px;
        background-color: #d1f2eb;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and cascade
@st.cache_resource
def load_resources():
        # Download model from Google Drive if not available
    if not os.path.exists("emotion_model.h5"):
        url = "https://drive.google.com/uc?id=1qp4GF5yV0kTu97OmPEkwr3AsIecTv1Et"
        gdown.download(url, "emotion_model.h5", quiet=False)
    model = tf.keras.models.load_model('emotion_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return model, face_cascade, classes

model, face_cascade, classes = load_resources()

# Sidebar
st.sidebar.title("🎭 Emotion AI")
st.sidebar.info("Upload a photo and let our AI detect the emotions! 🚀")
st.sidebar.markdown("---")
st.sidebar.subheader("How it works:")
st.sidebar.write("1. Upload an image (JPG/PNG).")
st.sidebar.write("2. AI detects faces using OpenCV.")
st.sidebar.write("3. CNN Model predicts the facial expression.")

# Main Header
st.markdown("<h1 style='text-align: center; color: #1f618d;'>Facial Emotion Recognition AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5dade2;'>A high-performance Computer Vision app for real-time sentiment analysis.</p>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Emotion Detection")
        
        # Preprocess for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            st.warning("No faces detected. Try another image!")
        else:
            # Create a copy for drawing
            result_img = img_array.copy()
            
            for (x, y, w, h) in faces:
                # Extract and predict
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                
                prediction = model.predict(roi_gray)
                maxindex = int(np.argmax(prediction))
                predicted_emotion = classes[maxindex]
                confidence = np.max(prediction) * 100
                
                # Draw on image
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 4)
                cv2.putText(result_img, f"{predicted_emotion} ({confidence:.1f}%)", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 3)
            
            st.image(result_img, use_container_width=True)
            
            # Display metrics for the first face detected
            if len(faces) > 0:
                st.markdown("### Primary Result")
                st.markdown(f"<div class='emotion-label'>{predicted_emotion.upper()} ({confidence:.1f}%)</div>", unsafe_allow_html=True)
                
                # Bar chart for probabilities
                st.markdown("---")
                st.subheader("Emotion Probability Distribution")
                prob_df = pd.DataFrame({
                    'Emotion': classes,
                    'Probability': prediction[0]
                })
                st.bar_chart(prob_df.set_index('Emotion'))

else:
    st.info("Please upload an image to start detection.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #85929e;'>Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
