import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
import glob
import time
import tkinter as tk
from tkinter import Label
import cv2

AUDIO_FILE = "captured_audio.wav"
VIDEO_FOLDER = "."  # Current folder

class VoiceToVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice to Animation Display")
        self.recognized_text = ""

        self.status_label = Label(root, text="Press 'Record' and speak.", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.record_button = tk.Button(root, text="🎤 Record Voice", command=self.record_voice)
        self.record_button.pack(pady=10)

        self.play_button = tk.Button(root, text="▶️ Play Video", command=self.play_videos, state=tk.DISABLED)
        self.play_button.pack(pady=10)

    def record_voice(self):
        self.status_label.config(text="Recording... Speak now!")
        self.root.update()

        duration = 5  # seconds
        samplerate = 16000

        try:
            audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
            sd.wait()

            wav.write(AUDIO_FILE, samplerate, audio)
            self.status_label.config(text="Recording complete. Recognizing speech...")
            self.root.after(500, self.recognize_speech)

        except Exception as e:
            self.status_label.config(text=f"Recording failed: {e}")

    def recognize_speech(self):
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = recognizer.record(source)

            self.recognized_text = recognizer.recognize_google(audio).lower()
            self.status_label.config(text=f"Recognized: {self.recognized_text}")
            self.play_button.config(state=tk.NORMAL)

        except sr.UnknownValueError:
            self.status_label.config(text="Couldn't understand speech.")
        except sr.RequestError as e:
            self.status_label.config(text=f"Speech service error: {e}")

    def play_videos(self):
        words = self.recognized_text.split()
        videos_to_play = find_matching_videos(words)

        if not videos_to_play:
            self.status_label.config(text="No matching videos found.")
            return

        self.status_label.config(text="Playing video(s)...")
        self.root.update()

        for video_path in videos_to_play:
            play_video(video_path)

        self.status_label.config(text="Playback finished.")

def find_matching_videos(words):
    """Return list of video files matching recognized words."""
    matched = []
    for word in words:
        pattern = os.path.join(VIDEO_FOLDER, f"{word}.mp4")
        matched.extend(glob.glob(pattern))
    return matched

def play_video(video_path):
    """Play video using OpenCV"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Animation Playback", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q' to stop
            break

    cap.release()
    cv2.destroyAllWindows()

# Start GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceToVideoApp(root)
    root.mainloop()
