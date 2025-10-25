import librosa
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# 1. Record audio input
# -----------------------------
def record_audio(duration=5, sr=22050):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    return np.squeeze(audio), sr

# -----------------------------
# 2. Extract audio features
# -----------------------------
def extract_audio_features(audio, sr):
    # Energy
    energy = np.mean(librosa.feature.rms(y=audio))
    # Pitch estimation using librosa.yin
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    # Spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    return np.array([pitch, energy, centroid])

# -----------------------------
# 3. Convert speech to text
# -----------------------------
def speech_to_text(audio_path="temp.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"🗣️ Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError:
            print("Speech Recognition service error.")
            return ""

# -----------------------------
# 4. Text-based emotion analysis
# -----------------------------
def analyze_text_emotion(text):
    if not text:
        return "neutral", 0.5
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    result = classifier(text)[0]
    label, score = result['label'].lower(), result['score']
    print(f"📘 Text Emotion: {label} (confidence: {score:.2f})")
    return label, score

# -----------------------------
# 5. Combine audio & text results
# -----------------------------
def combine_results(audio_feats, text_label, text_conf):
    # Simple rule-based fusion for demonstration
    pitch, energy, centroid = audio_feats
    if energy < 0.02 and pitch < 100:
        audio_emotion = "sad"
    elif pitch > 200 and energy > 0.05:
        audio_emotion = "happy"
    elif energy > 0.04 and pitch < 150:
        audio_emotion = "angry"
    else:
        audio_emotion = "neutral"

    print(f"🎧 Audio Emotion: {audio_emotion}")

    # Combine (weighted)
    if text_conf > 0.7:
        final_emotion = text_label
    else:
        # If uncertain, blend based on audio
        if audio_emotion == text_label:
            final_emotion = audio_emotion
        else:
            final_emotion = audio_emotion if energy > 0.03 else text_label

    print(f"💡 Final Detected Emotion: {final_emotion}")
    return final_emotion

# -----------------------------
# 6. Main program
# -----------------------------
if __name__ == "__main__":
    print("🎯 Emotion Detection (Audio + Text)")
    mode = input("Enter input type ('audio' or 'text'): ").strip().lower()

    if mode == "audio":
        audio, sr = record_audio()
        librosa.output.write_wav("temp.wav", audio, sr)
        audio_feats = extract_audio_features(audio, sr)
        text = speech_to_text("temp.wav")
        text_label, text_conf = analyze_text_emotion(text)
        combine_results(audio_feats, text_label, text_conf)

    elif mode == "text":
        text = input("Enter your text: ")
        analyze_text_emotion(text)

    else:
        print("Invalid input type. Choose 'audio' or 'text'.")
