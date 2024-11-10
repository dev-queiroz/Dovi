import os
import queue
import sounddevice as sd
import vosk
import json
import pyttsx3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

vosk_model_path = "models/vosk-model-pt"
model = vosk.Model(vosk_model_path)
recognizer = vosk.KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b")
model_ia = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b")
engine = pyttsx3.init()
engine.setProperty('rate', 250)

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
                As respostas devem ser claras e diretas, sem formatação especial ou símbolos.
                Pergunta: {text}"""
                
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model_ia.generate(inputs["input_ids"], max_length=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def speak(text):
    engine.say(text)
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

# Executa o programa
if __name__ == "__main__":
    main()
