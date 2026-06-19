from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import torchaudio
import torch
import numpy as np

import sys
audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "proyectos/transcripcion/ten.mp3"

initial_prompt = (
    "con objeto de, en virtud de, a efectos de, "
    "el Cabildo Insular, el Consejo de Gobierno, "
    "el consejero, la consejera, el presidente, la presidenta, "
    "el Pleno, el orden del día, el expediente, "
    "Tenerife, Gran Canaria, Fuerteventura, Lanzarote, Mogán, Arucas, Gáldar"
)

# --- 1. Transcripción ---
print("Transcribiendo...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
segments, info = model.transcribe(
    str(audio_path),
    language="es",
    initial_prompt=initial_prompt,
    beam_size=10,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=450,
        speech_pad_ms=200,
        threshold=0.7,
    ),
    condition_on_previous_text=False
)

all_segments = []
duration = info.duration
for segment in segments:
    if segment.no_speech_prob > 0.6:
        continue
    if segment.compression_ratio > 2.4:
        continue
    text = segment.text.strip()
    print(f"[{segment.end/duration*100:.1f}%] {text}")
    all_segments.append({"start": segment.start, "end": segment.end, "texto": text})

del model
print(f"Segmentos transcritos: {len(all_segments)}")

# --- 2. Embeddings de voz ---
print("Extrayendo huellas de voz...")
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)

wav, sr = torchaudio.load(str(audio_path))
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav.mean(dim=0)  # mono

embeddings = []
segmentos_validos = []
for seg in all_segments:
    chunk = wav[int(seg["start"] * 16000):int(seg["end"] * 16000)]
    if chunk.shape[0] < 8000:  # descartar segmentos de menos de 0.5s
        continue
    with torch.no_grad():
        emb = encoder.encode_batch(chunk.unsqueeze(0)).squeeze().cpu().numpy()
    embeddings.append(emb)
    segmentos_validos.append(seg)

print(f"Segmentos con huella válida: {len(segmentos_validos)}")

# --- 3. Clustering ---
if len(embeddings) > 1:
    # distance_threshold controla la sensibilidad:
    # más alto = menos oradores detectados, más bajo = más oradores
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.75,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(np.array(embeddings))
else:
    labels = [0] * len(embeddings)

n_oradores = len(set(labels))
print(f"Oradores detectados: {n_oradores}")

for i, seg in enumerate(segmentos_validos):
    seg["orador"] = f"ORADOR_{labels[i]+1:02d}"

# --- 4. Escribir resultado ---
output_path = audio_path.with_suffix(".txt")
with open(output_path, "w", encoding="utf-8") as f:
    orador_actual = None
    for seg in segmentos_validos:
        if seg["orador"] != orador_actual:
            f.write(f"\n[{seg['orador']}]\n")
            orador_actual = seg["orador"]
        f.write(seg["texto"] + "\n")

print(f"Guardado en: {output_path}")
