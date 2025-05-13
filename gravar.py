import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import torch
import librosa
import sounddevice as sd
import numpy as np
import keyboard
from threading import Event
from transformers import pipeline
from scipy.io import wavfile

# configs
SAMPLE_RATE = 44100
TARGET_SR = 16000
CHANNELS = 1
DTYPE = "int16"
SAVE_PATH = "audio_gravado.wav"

is_recording = False
audio_frames = []
stop_event = Event()


def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_frames.append(indata.copy())


def toggle_recording(e):
    global is_recording
    if not is_recording:
        print("\nGravação INICIADA...")
        audio_frames.clear()
        is_recording = True
    else:
        print("\nGravação PARADA.")
        is_recording = False
        stop_event.set()


keyboard.on_press_key("space", toggle_recording)

if __name__ == "__main__":
    print("Dispositivos de áudio disponíveis:")
    print(sd.query_devices())

    print("\nPressione ESPAÇO para começar/parar a gravação...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,

    ):
        stop_event.wait()

    if len(audio_frames) > 0:
        audio_array = np.concatenate(audio_frames, axis=0)
        wavfile.write(SAVE_PATH, SAMPLE_RATE, audio_array)
        print(f"Áudio salvo em: {SAVE_PATH}")


        audio, sr = librosa.load(SAVE_PATH, sr=SAMPLE_RATE, mono=True)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)


        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            device=device,
        )

        sample = {"array": audio_16k, "sampling_rate": TARGET_SR}
        prediction = pipe(sample, generate_kwargs={"language": "portuguese"})["text"]
        print(f"\nTranscrição: {prediction}")
    else:
        print("Nenhum áudio gravado.")
