import os
import cv2
import mediapipe as mp
import json
import numpy as np
import time
import pyttsx3  # Replaced subprocess with pyttsx3

# Ensure TensorFlow Lite warnings don't clutter logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Setting speech rate similar to espeak's 150

DATA_FILE = "gesture_templates.json"  # Ensure this file exists from training
DELAY_SECONDS = 5  # 5-second delay after speaking
BUFFER_SIZE = 5    # Number of frames to validate gesture recognition
THRESHOLD = 0.2    # Distance threshold for gesture matching

def speak_text(text):
    """Uses pyttsx3 to vocalize the detected word."""
    try:
        print(f"Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error with pyttsx3:", e)

def normalize_keypoints(keypoints):
    """Normalize keypoints for scale-invariant comparison."""
    keypoints = np.array(keypoints)
    min_vals = np.min(keypoints, axis=0)
    max_vals = np.max(keypoints, axis=0)
    denom = (max_vals - min_vals)
    denom[denom == 0] = 1e-6  # Avoid division by zero
    return ((keypoints - min_vals) / denom).flatten().tolist()

def load_templates():
    """Load pre-trained gesture templates."""
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        print("Loaded gestures:", list(data.keys()))
        return data
    except FileNotFoundError:
        print("No gesture data found! Train the system first.")
        exit(1)

def detect_gesture():
    """Main function for gesture detection and speech output."""
    templates = load_templates()
    cap = cv2.VideoCapture(0)
    last_detect_time = time.time()  # Initialize with current time
    speak = None  # Variable to hold the detected word
    gesture_buffer = []  # Buffer to store recent gesture detections

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        recognized_word = None
        best_distance = float("inf")

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                norm_keypoints = normalize_keypoints(keypoints)

                # Compare with saved gesture templates
                for word, template in templates.items():
                    distance = np.linalg.norm(np.array(norm_keypoints) - np.array(template))
                    if distance < best_distance:
                        best_distance = distance
                        recognized_word = word

                break  # Only process one hand per frame

        # Update the gesture buffer
        if recognized_word:
            gesture_buffer.append(recognized_word)
        else:
            gesture_buffer.append(None)

        if len(gesture_buffer) > BUFFER_SIZE:
            gesture_buffer.pop(0)

        # Confirm detection if all buffer values match
        if len(gesture_buffer) == BUFFER_SIZE and all(g == gesture_buffer[0] and g is not None for g in gesture_buffer):
            current_time = time.time()
            if current_time - last_detect_time > DELAY_SECONDS:
                print(f"Detected: {gesture_buffer[0]}")
                speak_text(gesture_buffer[0])
                last_detect_time = current_time
                gesture_buffer = []  # Reset buffer after detection

        # Stop speaking if no hand is detected
        if not results.multi_hand_landmarks and speak is not None:
            print("No hand detected. Stopping speech.")
            speak = None

        # Exit detection if ESC is pressed (useful when running without GUI)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()

if __name__ == "__main__":
    detect_gesture()
