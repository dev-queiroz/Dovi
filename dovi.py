import json
import pyaudio
import vosk
import pyttsx3
import requests

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

model = vosk.Model("models/vosk-model-pt")

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

API_URL = "http://127.0.0.1:1234/v1/chat/completions"

def listen_and_respond():
    print("Listening...")
    rec = vosk.KaldiRecognizer(model, 16000)
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text = json.loads(result)["text"]
            if text:
                print(f"You said: {text}")

                payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": text}]
                }
                response = requests.post(API_URL, json=payload)
                response_text = response.json()['choices'][0]['message']['content'].strip()

                print(f"Dovi says: {response_text}")
                speak(response_text)

listen_and_respond()
