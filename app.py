import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from transformers import pipeline
import tempfile
import scipy.io.wavfile as wav

# -----------------------------
# Helper Functions
# -----------------------------
def extract_audio_features(audio, sr):
    energy = np.mean(librosa.feature.rms(y=audio))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    return np.array([pitch, energy, centroid])

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except:
            return ""

@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_text_emotion(text):
    if not text:
        return "neutral", 0.0
    classifier = load_text_model()
    result = classifier(text)[0]
    return result['label'].lower(), result['score']

def combine_results(audio_feats, text_label, text_conf):
    pitch, energy, centroid = audio_feats
    if energy < 0.02 and pitch < 100:
        audio_emotion = "sad"
    elif pitch > 200 and energy > 0.05:
        audio_emotion = "happy"
    elif energy > 0.04 and pitch < 150:
        audio_emotion = "angry"
    else:
        audio_emotion = "neutral"

    if text_conf > 0.7:
        final_emotion = text_label
    else:
        final_emotion = audio_emotion if energy > 0.03 else text_label

    return audio_emotion, final_emotion

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎭 Emotion Detection from Audio + Text")

option = st.radio("Choose input type:", ["Text", "Audio Upload", "🎤 Live Audio"])

# ---------- TEXT MODE ----------
if option == "Text":
    text = st.text_input("Enter your text:")
    if st.button("Analyze Text"):
        emotion, conf = analyze_text_emotion(text)
        st.success(f"Detected Emotion: **{emotion}** (Confidence: {conf:.2f})")

# ---------- UPLOAD MODE ----------
elif option == "Audio Upload":
    st.info("Upload a short WAV audio file (5–10 s)")
    audio_file = st.file_uploader("Choose audio...", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file)
        audio, sr = librosa.load(audio_file, sr=22050)
        sf.write("temp.wav", audio, sr)

        audio_feats = extract_audio_features(audio, sr)
        text = speech_to_text("temp.wav")
        st.write("🗣️ Detected Speech:", text)

        text_label, text_conf = analyze_text_emotion(text)
        audio_emotion, final_emotion = combine_results(audio_feats, text_label, text_conf)

        st.write(f"🎧 Audio Emotion: **{audio_emotion}**")
        st.success(f"💡 Final Detected Emotion: **{final_emotion}**")

# ---------- LIVE AUDIO MODE ----------
else:
    st.info("Click below to record live audio 🎙️")
    duration = st.slider("Recording Duration (seconds)", 2, 10, 5)
    sample_rate = 22050

    if st.button("🎙️ Record"):
        st.warning("Recording... Speak now!")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        st.success("Recording complete!")

        # Save to a temporary file
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(tmpfile.name, sample_rate, (recording * 32767).astype(np.int16))
        st.audio(tmpfile.name, format="audio/wav")

        # Analyze recorded audio
        audio, sr = librosa.load(tmpfile.name, sr=22050)
        audio_feats = extract_audio_features(audio, sr)
        text = speech_to_text(tmpfile.name)
        st.write("🗣️ Detected Speech:", text)

        text_label, text_conf = analyze_text_emotion(text)
        audio_emotion, final_emotion = combine_results(audio_feats, text_label, text_conf)

        st.write(f"🎧 Audio Emotion: **{audio_emotion}**")
        st.success(f"💡 Final Detected Emotion: **{final_emotion}**")
