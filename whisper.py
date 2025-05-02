import torch
from transformers import pipeline
import librosa  # Para carregar áudios customizados

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Configurar pipeline (sem chunk_length_s ou batch_size)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device,
)

# Carregar áudio em PT-BR (exemplo local)
audio, sr = librosa.load("audio2.wav", sr=16000, mono=True)
sample = {"array": audio, "sampling_rate": sr}

# Transcrição forçando PT-BR
prediction = pipe(
    sample,
    generate_kwargs={"language": "portuguese", "task": "transcribe"}
)["text"]

print(f"Transcrição: {prediction}")
