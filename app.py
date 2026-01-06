import streamlit as st
import cv2
import time
import tempfile
import numpy as np

from ultralytics import YOLO
from gtts import gTTS
import whisper
import sounddevice as sd

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Vision Voice AI")

st.title("Vision Voice AI")
st.write("Live Object Detection with Voice + Speech Control")

# -------------------- Session State --------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

# -------------------- UI Buttons --------------------
col1, col2, col3 = st.columns(3)

with col1:
    start = st.button("Start Camera")
with col2:
    stop = st.button("Stop Camera")
with col3:
    listen = st.button("Listen (Voice Command)")

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

FRAME_WINDOW = st.empty()
AUDIO_PLACEHOLDER = st.empty()
TEXT_PLACEHOLDER = st.empty()

# -------------------- Models --------------------
yolo_model = YOLO("yolov8n.pt")

# Use BASE model (stable + smaller)
stt_model = whisper.load_model("base")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------------- Functions --------------------
def speak(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        AUDIO_PLACEHOLDER.audio(fp.name, format="audio/mp3")

def listen_command():
    fs = 16000
    duration = 5

    TEXT_PLACEHOLDER.info("Listening...")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = recording.flatten()

    # IMPORTANT: raw audio array → NO ffmpeg
    result = stt_model.transcribe(audio, fp16=False)
    return result["text"].lower()

# -------------------- Handle Voice Command --------------------
if listen:
    command = listen_command()
    TEXT_PLACEHOLDER.success(f"You said: {command}")

    if "stop" in command:
        st.session_state.run = False
        speak("Camera stopped")

    elif "start" in command:
        st.session_state.run = True
        speak("Camera started")

    elif "see" in command or "object" in command:
        speak("I am detecting objects around you")

# -------------------- Main Camera Loop --------------------
while st.session_state.run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not accessible")
        break

    results = yolo_model(frame, stream=True)
    detected = set()

    for result in results:
        frame = result.plot()
        for box in result.boxes:
            cls = int(box.cls[0])
            detected.add(yolo_model.names[cls])

    detected_text = ", ".join(detected)

    # Speak only when detection changes
    if detected_text and detected_text != st.session_state.last_spoken:
        speak(f"I see {detected_text}")
        st.session_state.last_spoken = detected_text

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    time.sleep(0.4)

camera.release()
