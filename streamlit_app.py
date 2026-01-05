import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter
import pyttsx3
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# UI Setup
st.set_page_config(page_title="ASL Interpreter", layout="centered")
st.title("ASL Interpreter")

# Load model
try:
    clf = joblib.load("models/static_sign_classifier.pkl")
    labels = clf.classes_
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# TTS Engine
try:
    engine = pyttsx3.init()
except Exception as e:
    st.warning(f"TTS Engine could not be initialized: {e}")
    engine = None

def speak(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

# Feature extractor
def extract_normalized_features(landmarks):
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    mid_finger_tip = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y, landmarks.landmark[12].z])
    norm_factor = np.linalg.norm(mid_finger_tip - wrist) + 1e-6
    features = []
    for i in range(1, 21):
        point = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z])
        dist = np.linalg.norm(point - wrist)
        normalized_dist = dist / norm_factor
        features.append(normalized_dist)
    return features

# Session State for prediction smoothing
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=10)
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = ""

# Controls
col1, col2 = st.columns(2)
with col1:
    run = st.checkbox('Start Camera', value=False)
with col2:
    enable_tts = st.checkbox("Enable Text-to-Speech", value=False)

FRAME_WINDOW = st.empty()
prediction_text = st.empty()

# Camera Loop
if run:
    # Try initializing webcam with different backends for macOS compatibility
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.warning("Camera 0 failed to open. Trying index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Could not open any webcam. Please ensure you have granted camera permissions to your terminal/browser and no other app is using the camera.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam. If you are on macOS, you may need to grant permission to your terminal (System Settings > Privacy & Security > Camera).")
                break
            
            # Mirror and Convert
            frame = cv2.flip(frame, 1)
            # Check if frame is valid
            if frame is None:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            result = hands.process(rgb)
            
            predicted_label = "No Hand"
            confidence = 0.0
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    try:
                        features = extract_normalized_features(hand_landmarks)
                        features = np.array(features).reshape(1, -1)
                        
                        probs = clf.predict_proba(features)[0]
                        max_idx = np.argmax(probs)
                        predicted_label = labels[max_idx]
                        confidence = probs[max_idx]
                    except Exception as e:
                        # Suppress transient errors
                        pass

            # Smoothing
            st.session_state.history.append(predicted_label)
            if st.session_state.history:
                smoothed_prediction = Counter(st.session_state.history).most_common(1)[0][0]
            else:
                smoothed_prediction = "No Prediction"

            # Display on Frame
            # Increased font scale (3.0 and 2.0) and thickness (5 and 3) for better visibility
            cv2.putText(frame, f"Pred: {smoothed_prediction}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 5)
            cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            
            # Update Streamlit Image
            # Convert BGR back to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            prediction_text.markdown(f"### Current Prediction: **{smoothed_prediction}**")

            # TTS Output
            if enable_tts and smoothed_prediction != st.session_state.last_spoken and smoothed_prediction not in ["No Hand", "No Prediction"]:
                speak(smoothed_prediction)
                st.session_state.last_spoken = smoothed_prediction
                
            # Stop button logic handled by checkbox state
            if not run:
                break
        
        cap.release()

