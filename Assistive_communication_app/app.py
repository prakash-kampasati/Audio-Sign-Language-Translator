import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from gtts import gTTS
import pygame
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr

# Initialize pygame mixer for stable audio playback
pygame.mixer.init()

# ================= CONFIG =================
DATA_FILE = "gesture_templates.json"
AUDIO_DIR = "gesture_audio"  # NEW: Folder to store individual word sounds
ANIMATION_FOLDER = "animation"
AUDIO_FILE = "captured_audio.wav"

SENTENCE_VIDEO_FOLDER = "sentence_videos"
SENTENCE_VIDEO_PATH = os.path.join(SENTENCE_VIDEO_FOLDER, "last_sentence.mp4")

TARGET_RADIUS = 140
MATCH_THRESHOLD = 0.75  # INCREASED: Helps recognize multiple signs easily
BUFFER_SIZE = 3
SPEAK_DELAY = 1.2  # Reduced delay for smoother multiple sign detection

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
os.makedirs(SENTENCE_VIDEO_FOLDER, exist_ok=True)

# ================= INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def show_info(title, msg):
    temp = tk.Tk()
    temp.attributes("-topmost", True)
    temp.withdraw()
    messagebox.showinfo(title, msg, parent=temp)
    temp.destroy()

def show_error(title, msg):
    temp = tk.Tk()
    temp.attributes("-topmost", True)
    temp.withdraw()
    messagebox.showerror(title, msg, parent=temp)
    temp.destroy()

# ================= UTILS =================
def normalize_keypoints(points):
    pts = np.array(points)
    min_v = pts.min(axis=0)
    max_v = pts.max(axis=0)
    denom = max_v - min_v
    denom[denom == 0] = 1e-6
    return ((pts - min_v) / denom).flatten()

def draw_target(frame):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    cv2.circle(frame, center, TARGET_RADIUS, (0, 255, 0), 2)
    return center

def hand_inside_target(hand, center, shape):
    h, w, _ = shape
    wrist = hand.landmark[0]
    x, y = int(wrist.x * w), int(wrist.y * h)
    return np.sqrt((x - center[0])**2 + (y - center[1])**2) < TARGET_RADIUS

# ================= TRAIN (With Auto-Audio Saving) =================
def train_gesture():
    root = tk.Tk()
    root.attributes("-topmost", True) # Force the 'Enter Word' box to the front
    root.withdraw()
    root.lift()
    root.focus_force()
    
    word = simpledialog.askstring("Train Gesture", "Enter word name:", parent=root)
    if not word:
        root.destroy()
        return

    # STEP 1: Save the audio of the word immediately using gTTS
    try:
        clean_word = word.lower().strip()
        audio_path = os.path.join(AUDIO_DIR, f"{clean_word}.mp3")
        
        # Only generate if it doesn't already exist
        if not os.path.exists(audio_path):
            tts = gTTS(text=word, lang='en')
            tts.save(audio_path)
    except Exception as e:
        print(f"Audio saving failed: {e}")

    # STEP 2: Capture Hand Data
    cap = cv2.VideoCapture(0)
    samples = []
    messagebox.showinfo("Training", f"Show sign for '{word}' inside the circle", parent=root)

    while len(samples) < 60:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        center = draw_target(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                if hand_inside_target(hand, center, frame.shape):
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                    samples.append(normalize_keypoints(pts))
                    cv2.putText(frame, f"Capturing: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Training Gesture", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

    if samples:
        template = np.mean(samples, axis=0).tolist()
        data = {}
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f: data = json.load(f)
        
        data[word.lower().strip()] = template
        with open(DATA_FILE, "w") as f: json.dump(data, f, indent=4)
        show_info("Success", f"'{word}' saved with audio!")

# ================= SIGN TO AUDIO (Play from Folder) =================
def sign_to_audio_loop():
    global stop_camera
    stop_camera = False

    if not os.path.exists(DATA_FILE):
        show_error("Error", "Train gestures first")
        return

    with open(DATA_FILE) as f: templates = json.load(f)

    cap = cv2.VideoCapture(0)
    buffer = []
    last_spoken = time.time()
    last_word = ""

    cv2.namedWindow("Sign to Audio", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Sign to Audio", cv2.WND_PROP_TOPMOST, 1)

    while cap.isOpened() and not stop_camera:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        center = draw_target(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        detected = None
        best_score = 999.0

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                if hand_inside_target(hand, center, frame.shape):
                    pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                    norm = normalize_keypoints(pts)
                    for word, temp in templates.items():
                        distance = np.linalg.norm(norm - np.array(temp))
                        if distance < best_score:
                            best_score = distance
                            detected = word
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if detected and best_score < MATCH_THRESHOLD:
            buffer.append(detected)
        else:
            buffer.append(None)
        
        if len(buffer) > BUFFER_SIZE: buffer.pop(0)

        # Sequential Audio Playback
        if len(buffer) == BUFFER_SIZE and all(x == buffer[0] for x in buffer) and buffer[0]:
            current_word = buffer[0]
            if (current_word != last_word) or (time.time() - last_spoken > SPEAK_DELAY):
                audio_file = os.path.join(AUDIO_DIR, f"{current_word.lower()}.mp3")
                if os.path.exists(audio_file):
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                
                last_spoken = time.time()
                last_word = current_word
                buffer.clear()

        cv2.putText(frame, f"Detected: {detected}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sign to Audio", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

def start_sign_to_audio():
     sign_to_audio_loop()

# ================= REMAINING UTILS (Video/Audio Sign) =================
def play_animation(path, play_time=10):
    cap = cv2.VideoCapture(path)
    win_name = "Sentence Video"
    
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        last_frame = frame.copy() # Keep a copy of the very last frame
        cv2.imshow(win_name, frame)
        
        # Use waitKey(30) for normal human-readable speed
        if cv2.waitKey(30) & 0xFF == 27:
            cap.release()
            cv2.destroyWindow(win_name)
            return

    # NEW: Keep the last frame on screen for 2 seconds before closing
    if last_frame is not None:
        cv2.imshow(win_name, last_frame)
        cv2.waitKey(2000) # Wait 2000ms (2 seconds)

    cap.release()
    cv2.destroyWindow(win_name)

def save_sentence_video(words, out_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    # Increase FPS (e.g., to 30) if your source videos are high-speed
    fps = 30 

    for word in words:
        mp4_path = os.path.join(ANIMATION_FOLDER, f"{word.lower()}.mp4")
        if not os.path.exists(mp4_path): 
            continue
            
        cap = cv2.VideoCapture(mp4_path)
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            if out is None:
                h, w, _ = frame.shape
                out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            out.write(frame)
        cap.release()
        
    if out: 
        out.release()

# def save_sentence_video(words, out_path):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = None
#     for word in words:
#         mp4_path = os.path.join(ANIMATION_FOLDER, f"{word}.mp4")
#         if not os.path.exists(mp4_path): continue
#         cap = cv2.VideoCapture(mp4_path)
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             if out is None:
#                 h, w, _ = frame.shape
#                 out = cv2.VideoWriter(out_path, fourcc, 25, (w, h))
#             out.write(frame)
#         cap.release()
#     if out: out.release()

def audio_to_sign():
    duration, samplerate = 6, 16000
    show_info("Speak", "Recording for 6 seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(AUDIO_FILE, samplerate, audio)
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as src:
        data = r.record(src)
    try:
        text = r.recognize_google(data).lower()
        save_sentence_video(text.split(), SENTENCE_VIDEO_PATH)
        play_saved_sentence()
    except: show_error("Error", "Speech not recognized")

def play_saved_sentence():
    if os.path.exists(SENTENCE_VIDEO_PATH): play_animation(SENTENCE_VIDEO_PATH)
    else: show_error("Error", "No saved video")

def start_desktop_gui():
    root = tk.Tk()
    root.title("Assistive Communication System")
    root.geometry("420x450")
    tk.Label(root, text="Sign Language Interface", font=("Arial", 14, "bold")).pack(pady=20)
    tk.Button(root, text="Train Gesture", width=30, command=train_gesture).pack(pady=5)
    tk.Button(root, text="Sign to Audio", width=30, command=start_sign_to_audio).pack(pady=5)
    tk.Button(root, text="Audio to Sign", width=30, command=audio_to_sign).pack(pady=5)
    tk.Button(root, text="Exit", width=30, command=root.destroy).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    import sys
    
    # Check if a command was sent from the web dashboard
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--train":
            train_gesture()
        elif command == "--recognition":
            start_sign_to_audio()
        elif command == "--audio":
            audio_to_sign()
        elif command == "--play":
            play_saved_sentence()
    else:
        # If you just run app.py normally, it opens the desktop menu
        start_desktop_gui()