# ASL Alphabet Interpreter

A computer vision application that recognizes American Sign Language (ASL) alphabets in real-time. Built with Python, it uses MediaPipe for hand tracking and a Random Forest classifier to interpret gestures into text.

## Overview

I built this project to experiment with gesture recognition and accessible interfaces. It captures video from your webcam, tracks hand landmarks, and classifies them into letters (A-Z). It also features a Text-to-Speech (TTS) engine that can read out the recognized letters, making it a functional tool for two-way communication.

## How It Works

The core pipeline consists of three stages:
1. **Hand Tracking**: Uses Google's MediaPipe to extract 21 3D landmarks from your hand.
2. **Feature Extraction**: Calculates the relative distances between landmarks to create a normalized feature vector (making it scale-invariant).
3. **Classification**: Passes these features through a trained Random Forest model (`scikit-learn`) to predict the letter.

## Getting Started

### Prerequisites
You'll need **Python 3.8** (or a compatible environment).

### Installation


1. Clone the repo:
   ```bash
   git clone [https://github.com/NeelMaddu268/ASL-Interpreter.git](https://github.com/NeelMaddu268/ASL_Interpreter.git)
   cd asl-interpreter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

The model will be loaded automatically (it is light enough to be included in the repo).

## Controls
- **Start Webcam**: Toggles the camera feed.
- **Enable Text-to-Speech**: If checked, the app will speak the detected letter when it changes. 

## Dataset
I used the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle for training. The model was trained on approx. 59,000 samples.

## License
MIT
