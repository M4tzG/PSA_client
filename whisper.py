import torch
from transformers import pipeline
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device,
)

audio, sr = librosa.load("audio2.wav", sr=16000, mono=True)
sample = {"array": audio, "sampling_rate": sr}

prediction = pipe(
    sample,
    generate_kwargs={"language": "portuguese", "task": "transcribe"}
)["text"]

print(f"Transcrição: {prediction}")
