import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# -----------------------------
# Helper Functions
# -----------------------------
def extract_audio_features(audio, sr):
    energy = np.mean(librosa.feature.rms(y=audio))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    return np.array([pitch, energy, centroid])

@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_speech_to_text_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def speech_to_text(audio_path):
    asr = load_speech_to_text_model()
    try:
        result = asr(audio_path)
        return result["text"].strip()
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        return ""

def analyze_text_emotion(text):
    if not text.strip():
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
st.set_page_config(page_title="🎭 Emotion Detector", layout="centered")
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
    st.info("Upload a short WAV audio file (5–10 seconds)")
    audio_file = st.file_uploader("Choose audio...", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file)
        audio, sr = librosa.load(audio_file, sr=22050)
        sf.write("temp.wav", audio, sr)

        audio_feats = extract_audio_features(audio, sr)
        text = speech_to_text("temp.wav")
        st.write("🗣️ Detected Speech:", text or "(No speech detected)")

        text_label, text_conf = analyze_text_emotion(text)
        audio_emotion, final_emotion = combine_results(audio_feats, text_label, text_conf)

        st.write(f"🎧 Audio Emotion: **{audio_emotion}**")
        st.success(f"💡 Final Detected Emotion: **{final_emotion}**")

# ---------- LIVE AUDIO MODE ----------
else:
    st.info("Click below to record your voice 🎙️")

    # Setup WebRTC
    webrtc_ctx = webrtc_streamer(
        key="live-audio",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False}
        ),
        async_processing=True
    )

    if webrtc_ctx.state.playing:
        st.write("Recording... Speak now!")
        if st.button("Stop & Analyze"):
            # Retrieve recorded audio frames
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1.0)
            
            if not audio_frames:
                st.error("No audio detected. Please record again.")
            else:
                # Convert frames to numpy array
                audio_np = np.concatenate([f.to_ndarray() for f in audio_frames], axis=0)
                
                # Save to temporary WAV file
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tmpfile.name, audio_np, 44100)
                st.audio(tmpfile.name, format="audio/wav")

                # Analyze audio
                audio, sr = librosa.load(tmpfile.name, sr=22050)
                audio_feats = extract_audio_features(audio, sr)
                text = speech_to_text(tmpfile.name)
                st.write("🗣️ Detected Speech:", text or "(No speech detected)")

                text_label, text_conf = analyze_text_emotion(text)
                audio_emotion, final_emotion = combine_results(audio_feats, text_label, text_conf)

                st.write(f"🎧 Audio Emotion: **{audio_emotion}**")
                st.success(f"💡 Final Detected Emotion: **{final_emotion}**")
