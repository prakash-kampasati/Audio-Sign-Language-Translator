import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow Lite warnings

import cv2
import mediapipe as mp
import json
import numpy as np
import time
import tkinter as tk
from tkinter import simpledialog

# Set lower camera resolution for Raspberry Pi 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATA_FILE = "gesture_templates.json"
SAMPLES_PER_GESTURE = 130  # Number of samples to collect for each gesture

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints)
    min_vals = np.min(keypoints, axis=0)
    max_vals = np.max(keypoints, axis=0)
    denom = (max_vals - min_vals)
    # Prevent division by zero
    denom[denom == 0] = 1e-6
    normed = ((keypoints - min_vals) / denom).flatten().tolist()
    return normed

def collect_samples(word):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    samples = []
    count = 0
    print(f"Collecting {SAMPLES_PER_GESTURE} samples for gesture: '{word}'")
    
    while cap.isOpened() and count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                norm_keypoints = normalize_keypoints(keypoints)
                samples.append(norm_keypoints)
                count += 1
                cv2.putText(frame, f"Sample {count}/{SAMPLES_PER_GESTURE}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break  # Process one hand per frame
        
        cv2.imshow("Training", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit early
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return samples

def average_template(samples):
    # Compute the average (mean) of all samples to form a robust template
    arr = np.array(samples)
    avg = np.mean(arr, axis=0).tolist()
    return avg

def update_templates(word, template):
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    
    data[word] = template
    
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Template for '{word}' saved.")

def training_main():
    root = tk.Tk()
    root.withdraw()
    word = simpledialog.askstring("Input", "Enter a word for the gesture:")
    if not word:
        return
    samples = collect_samples(word)
    if samples:
        template = average_template(samples)
        update_templates(word, template)
    else:
        print("No samples were collected.")

if __name__ == "__main__":
    training_main()
