from flask import Flask, render_template, redirect, url_for
import subprocess
import sys

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train")
def train():
    # Launches app.py as a separate program so the window pops up immediately
    subprocess.Popen([sys.executable, "app.py", "--train"])
    return redirect(url_for("index"))

@app.route("/sign_to_audio")
def sign_to_audio():
    subprocess.Popen([sys.executable, "app.py", "--recognition"])
    return redirect(url_for("index"))

@app.route("/audio_to_sign")
def audio_to_sign_route():
    subprocess.Popen([sys.executable, "app.py", "--audio"])
    return redirect(url_for("index"))

@app.route("/play_saved")
def play_saved():
    subprocess.Popen([sys.executable, "app.py", "--play"])
    return redirect(url_for("index"))

if __name__ == "__main__":
    # debug=False is mandatory for this to work correctly
    app.run(debug=False, port=5000)