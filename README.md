# Alertify 🚗 - Driver Drowsiness and Attention Detection

**Alertify** is an advanced AI-powered application designed to enhance road safety by detecting driver drowsiness and inattention in real time. Using computer vision and machine learning techniques, it monitors eye closure, yawning, and head tilt to provide instant alerts, ensuring drivers stay attentive.

---

## 🌟 Key Features
- **Eye Aspect Ratio (EAR):** Detects prolonged eye closure, a key indicator of drowsiness.
- **Mouth Aspect Ratio (MAR):** Identifies yawning patterns, which often signal fatigue.
- **Head Tilt Detection:** Monitors unusual head tilt angles that indicate inattention or distraction.
- **Real-Time Alerts:** Plays an alarm sound and highlights warnings in the interface for quick action.
- **Customizable Thresholds:** Allows users to configure detection sensitivity via an interactive sidebar.
- **Offline Support:** Works without an internet connection after the initial setup.
- **Video Feed Integration:** Uses a webcam for real-time monitoring.

---

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **Streamlit**: For the user interface and visualization.
- **OpenCV**: Real-time image and video processing.
- **Dlib**: Facial landmark detection.
- **Mediapipe**: Face mesh detection for head pose estimation.
- **Numpy & Scipy**: Mathematical operations for aspect ratio calculations.
- **gTTS & Playsound**: Generates and plays audio alerts.

---

## 🚀 How It Works

### 1. **Input Data:**
   - **Video Feed:** Captures a live video stream from the webcam.
   - **Facial Landmarks:** Dlib and Mediapipe detect facial landmarks, such as eyes, mouth, and head positions.

### 2. **Detection Algorithms:**
   - **Eye Aspect Ratio (EAR):**
     - Calculates the distance between vertical and horizontal eye landmarks.
     - Triggers an alarm if eyes remain closed beyond the configured threshold.
   - **Mouth Aspect Ratio (MAR):**
     - Measures the aspect ratio of the mouth to detect yawning.
     - Triggers an alert for prolonged yawning.
   - **Head Pose Estimation:**
     - Uses head pose angles to detect distractions like looking away from the road.
     - Alerts the driver if head tilt exceeds a safe angle.

### 3. **Alerts:**
   - **Visual Alert:** Background turns red, and warnings are displayed in the app.
   - **Audio Alert:** An alarm sound plays to regain driver attention.

---

## 📷 Screenshots
![Driver Home](https://github.com/user-attachments/assets/6aa381ab-a29f-4e65-9bd7-c02bc7680c63)

⚙️ Configuration Options
Sidebar Controls:
Eye Aspect Ratio Threshold (EAR): Configure sensitivity for eye closure detection.
Mouth Aspect Ratio Threshold (MAR): Set the yawning detection threshold.
Head Tilt Alarm Trigger: Adjust the duration for head tilt detection before triggering an alert.
Language Settings: Select a language for warning messages (e.g., English, Spanish, etc.).
Resolution Options: Choose from 640x480, 1280x720, or 1920x1080 for video feed quality.

📊 Algorithms
Eye Aspect Ratio (EAR):

Formula:
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
Monitors eye closure using vertical and horizontal distances between landmarks.

Mouth Aspect Ratio (MAR):

Formula:
MAR = (||p3 - p9|| + ||p5 - p7||) / (2 * ||p1 - p4||)
Measures mouth opening to detect yawning patterns.
Head Pose Estimation:

Uses 3D model points of facial landmarks to calculate head tilt angles.

