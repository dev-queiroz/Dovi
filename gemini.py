import os
import queue
import sounddevice as sd
import vosk
import json
import pyttsx3
import google.generativeai as genai
import sys
import re

genai.configure(api_key="AIzaSyA7PZX3s5FgNeRGHXv2CwZ6si98mBc5LM4")
model = genai.GenerativeModel("gemini-1.5-flash")
vosk_model_path = "models/vosk-model-pt"
v_model = vosk.Model(vosk_model_path)
recognizer = vosk.KaldiRecognizer(v_model, 16000)
audio_queue = queue.Queue()

engine = pyttsx3.init()
engine.setProperty('rate', 250)

generation_config = {
  "temperature": 0.55,
  "top_p": 0.5,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config, # type: ignore
)

chat_session = model.start_chat(
  history=[ ]
)

if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"Modelo Vosk não encontrado em '{vosk_model_path}'.")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))

def recognize_speech():
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            return result.get("text", "")

def generate_response(text):
    prompt = f"""Você é Dovi, um assistente de voz criado por Douglas Queiroz. 
                Responda sempre de forma natural e amigável, como se estivesse em uma conversa cotidiana.
                Você NUNCA deve usar emojis, "😄" ou caracteres especiais como '*', '#'. 
                As respostas devem ser claras e diretas, sem formatação especial ou símbolos.
                Pergunta: {text}"""
    
    response = model.generate_content(prompt)
    if response and response._result.candidates:
        return response._result.candidates[0].content.parts[0].text
    return "Desculpe, não consegui entender sua pergunta."

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def speak(text):
    clean_text = remove_emojis(text)
    engine.say(clean_text)
    engine.runAndWait()

def main():
    speak("Olá humanos, eu sou a Dovi. Como vocês estão?")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=audio_callback):
        while True:
            print("Escutando...")
            text = recognize_speech()
            if text:
                print("Você disse:", text)

                if "tchau" in text.lower():
                    speak("Tchau! Até mais!")
                    sys.exit()

                response = generate_response(text)
                print("Resposta:", response)

                speak(response)

if __name__ == "__main__":
    main()
