import streamlit as st
import cv2
import numpy as np
import time
import threading
from playsound import playsound
from gtts import gTTS
import os
import mediapipe as mp
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import logging

# Configure logging
logging.basicConfig(filename="driver_drowsiness_detection.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Configuration for Streamlit
st.title("Alertify - 🚗 Driver Drowsiness and Attention Detection")
st.markdown("""
This app uses computer vision to detect drowsiness or inattention by analyzing:
- Eye Aspect Ratio (EAR) for eye closure
- Mouth Aspect Ratio (MAR) for yawning
- Head tilt angle
""")

# Sidebar for file inputs and settings
shape_predictor_path = st.sidebar.text_input("Path to Shape Predictor", "/Users/shivanshinigam/shape_predictor_68_face_landmarks.dat")
alarm_path = st.sidebar.text_input("Path to Alarm Sound", "/Users/shivanshinigam/Desktop/alarm.wav")
language = st.sidebar.selectbox("Select Language for Warning", ['en', 'es', 'fr', 'de'])

# Threshold Settings
EYE_AR_THRESH = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.2, 0.3, 0.25, 0.01)
MOUTH_AR_THRESH = st.sidebar.slider("Mouth Aspect Ratio Threshold", 0.7, 1.0, 0.79, 0.01)
EYE_CLOSED_TIME = st.sidebar.slider("Eye Closure Alarm Trigger (seconds)", 1, 10, 5)
HEAD_TILT_TIME = st.sidebar.slider("Head Tilt Alarm Trigger (seconds)", 1, 10, 5)

# Sidebar option for video resolution
resolution = st.sidebar.radio("Select Video Resolution", ["640x480", "1280x720", "1920x1080"])

# Global Variables for Alarm Control
is_alarm_playing = False
stop_alarm_flag = False

# Initialize the start times for various checks
eye_closed_start_time = None
head_tilt_start_time = None  # Initialize this variable to prevent the error
yawn_start_time = None


# Function to play alarm and stop manually
def play_alarm(alarm_path):
    global is_alarm_playing, stop_alarm_flag
    if not is_alarm_playing:
        is_alarm_playing = True
        stop_alarm_flag = False
        threading.Thread(target=playsound_with_stop, args=(alarm_path,)).start()


# Function to play sound with manual stop
def playsound_with_stop(alarm_path):
    global is_alarm_playing, stop_alarm_flag
    try:
        while not stop_alarm_flag:
            playsound(alarm_path)
    except Exception as e:
        st.error(f"Error playing alarm: {e}")
    is_alarm_playing = False


# Stop Alarm Button
if st.sidebar.button("Stop Alarm"):
    stop_alarm_flag = True
    is_alarm_playing = False  # Make sure to update the status of alarm playing


# Detect drowsiness and distractions
def detect_drowsiness(frame):
    global eye_closed_start_time, head_tilt_start_time, yawn_start_time

    # Resize frame and convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eyes
        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # Eye Closure Detection
        if ear < EYE_AR_THRESH:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elif (time.time() - eye_closed_start_time) > EYE_CLOSED_TIME:
                play_alarm(alarm_path)
                eye_closed_start_time = None
                st.markdown('<style>body {background-color: red;}</style>', unsafe_allow_html=True)  # Red background
                st.warning("⚠️ **Warning:** Eye Closure Detected! Please stay alert.", icon="⚠️")
                logging.warning("Eye closure detected!")
                return frame  # Skip further processing during warning
        else:
            eye_closed_start_time = None

        # Mouth Detection (Yawning)
        mouth = shape[49:68]
        mouthMAR = mouth_aspect_ratio(mouth)
        if mouthMAR > MOUTH_AR_THRESH:
            if yawn_start_time is None:
                yawn_start_time = time.time()
            elif (time.time() - yawn_start_time) > 5:
                play_alarm(alarm_path)
                yawn_start_time = None
                st.markdown('<style>body {background-color: red;}</style>', unsafe_allow_html=True) 
                st.warning("⚠️ **Warning:** Yawning Detected! Please stay alert.", icon="⚠️")
                logging.warning("Yawning detected!")
                return frame  # Skip further processing during warning
        else:
            yawn_start_time = None

        # Head Tilt Detection
        image_points = np.array([
            (shape[33][0], shape[33][1]),  # Nose tip
            (shape[8][0], shape[8][1]),    # Chin
            (shape[36][0], shape[36][1]),  # Left eye corner
            (shape[45][0], shape[45][1]),  # Right eye corner
            (shape[48][0], shape[48][1]),  # Left mouth corner
            (shape[54][0], shape[54][1])   # Right mouth corner
        ], dtype="double")

        (head_tilt_degree, _, _, _) = getHeadTiltAndCoords(frame.shape[:2], image_points, frame.shape[0])
        if head_tilt_degree and abs(head_tilt_degree[0]) > 10:
            if head_tilt_start_time is None:
                head_tilt_start_time = time.time()
            elif (time.time() - head_tilt_start_time) > HEAD_TILT_TIME:
                play_alarm(alarm_path)
                head_tilt_start_time = None
                st.markdown('<style>body {background-color: red;}</style>', unsafe_allow_html=True)  
                st.warning("⚠️ **Warning:** Head Tilt Detected! Please stay alert.", icon="⚠️")
                logging.warning("Head tilt detected!")
                return frame  # Skip further processing during warning
        else:
            head_tilt_start_time = None

    return frame


# Initialize dlib and MediaPipe
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

# Video Capture for live feed
FRAME_WINDOW = st.image([])

# Load the image for driver alertness concept
st.image("https://i.ytimg.com/vi/5QECtDjr_f4/maxresdefault.jpg", width=1000)

if st.button("Start Detection"):
    # Set video resolution
    resolutions = {
        "640x480": (640, 480),
        "1280x720": (1280, 720),
        "1920x1080": (1920, 1080)
    }
    
    width, height = resolutions[resolution]
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_drowsiness(frame)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
