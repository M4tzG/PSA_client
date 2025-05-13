import torch
import librosa
from scipy.io import wavfile
from transformers import pipeline
from speechbrain.pretrained import SepformerSeparation as Separator
from speechbrain.pretrained import EncoderClassifier

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def separar_vozes(audio_path, output_path="voz_isolada.wav"):
    model = Separator.from_hparams(
        source="speechbrain/sepformer-wham",
        savedir="pretrained_models/sepformer-wham",
        run_opts={"device": device}
    )

    est_sources = model.separate_file(audio_path)

    wavfile.write(output_path, 16000, est_sources[:, :, 0].detach().cpu().numpy().squeeze())
    return output_path

def criar_perfil_voz(audio_limpo_path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    audio, sr = librosa.load(audio_limpo_path, sr=16000, mono=True)
    embedding = classifier.encode_batch(torch.tensor(audio).unsqueeze(0).to(device))
    return embedding

def identificar_voz(audio_isolado_path, embedding_referencia):
    audio, sr = librosa.load(audio_isolado_path, sr=16000, mono=True)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    embedding_teste = classifier.encode_batch(torch.tensor(audio).unsqueeze(0).to(device))

    similaridade = torch.nn.functional.cosine_similarity(embedding_referencia, embedding_teste, dim=2)
    return similaridade.item()

# --- Execução Principal ---
if __name__ == "__main__":
    # 1. separar vozes do audio
    voz_isolada_path = separar_vozes("audio_gravado.wav")

    # 2. carregar perfil vocal (meh)
    seu_perfil = criar_perfil_voz("minha_voz.wav")

    # 3. verifica voz isolada
    similaridade = identificar_voz(voz_isolada_path, seu_perfil)
    print(similaridade)
    if similaridade > 0:
        # 4. transcreve
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium",
            device=device,
        )
        audio, sr = librosa.load(voz_isolada_path, sr=16000, mono=True)
        sample = {"array": audio, "sampling_rate": sr}
        prediction = pipe(sample, generate_kwargs={"language": "portuguese"})["text"]
        print(f"Sua voz isolada: {prediction}")
    else:
        print("Voz não identificada. Ajuste o modelo ou threshold.")
