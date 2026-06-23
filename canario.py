from faster_whisper import WhisperModel
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel
from spylls.hunspell import Dictionary
from rapidfuzz import fuzz, process
from speechbrain.inference.speaker import EncoderClassifier
import librosa
import numpy as np
import torch
import re
import json
import unicodedata

dic = Dictionary.from_files('/usr/share/hunspell/es_ES')

PATRONES_CAMBIO = [
    r'\btiene\s+la\s+palabra\b',
]

CAMBIO_THRESHOLD = 0.3  # con embeddings neuronales: mismo orador ~0.05-0.2, distinto ~0.3-0.7
SILENCIO_MINIMO = 0.5    # segundos de pausa para considerar posible cambio de orador
POST_VENTANA = 5.0       # segundos de audio tras el silencio para el embedding del nuevo orador
MAX_PRE_VENTANA = 30.0   # máximo de segundos antes del silencio (cap para turnos muy largos)
MIN_PRE_DURACION = 2.0   # mínimo de segundos de audio previo para que la comparación sea fiable

def capitalizar_oraciones(texto):
    texto = re.sub(r'(\.\s+)([a-záéíóúüñ])', lambda m: m.group(1) + m.group(2).upper(), texto)
    if texto:
        texto = texto[0].upper() + texto[1:]
    return texto

def normalizar(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()

def cargar_indice_municipios(json_path):
    with open(json_path) as f:
        data = json.load(f)
    municipios = []
    for isla in data:
        municipios.extend(isla["municipios"])
    return [(normalizar(m).replace(" ", ""), m) for m in municipios]

def sugerir_correcciones(txt_path, desconocidas, indice, threshold=80):
    choices = [x[0] for x in indice]
    nombres = [x[1] for x in indice]
    lineas = []
    for palabra in desconocidas:
        palabra_norm = normalizar(palabra).replace(" ", "")
        if len(palabra_norm) < 4:
            lineas.append(palabra)
            continue
        matches = process.extract(palabra_norm, choices, scorer=fuzz.ratio, limit=2)
        if matches and matches[0][1] >= threshold:
            sugerencias = ", ".join(f"{nombres[m[2]]} ({m[1]:.0f}%)" for m in matches)
            lineas.append(f"{palabra} → {sugerencias}")
        else:
            lineas.append(palabra)
    output = Path(txt_path).with_stem(Path(txt_path).stem + "_desconocidas")
    output.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Sugerencias guardadas en {output}")

def aplicar_correcciones(txt_path, desconocidas, indice, threshold=70):
    choices = [x[0] for x in indice]
    nombres = [x[1] for x in indice]
    texto = Path(txt_path).read_text(encoding="utf-8")
    aplicadas = 0
    for palabra in desconocidas:
        palabra_norm = normalizar(palabra).replace(" ", "")
        if len(palabra_norm) < 4:
            continue
        match = process.extractOne(palabra_norm, choices, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            correccion = nombres[match[2]]
            texto = re.sub(r'\b' + re.escape(palabra) + r'\b', correccion, texto, flags=re.IGNORECASE)
            aplicadas += 1
            print(f"  '{palabra}' → '{correccion}' ({match[1]:.0f}%)")
    Path(txt_path).write_text(texto, encoding="utf-8")
    print(f"{aplicadas} correcciones aplicadas en {txt_path}")

def es_cambio_de_orador(texto):
    texto_norm = normalizar(texto)
    return any(re.search(p, texto_norm) for p in PATRONES_CAMBIO)

def split_en_cambio(texto):
    """Parte el texto en (antes_del_cambio, despues_del_cambio) si contiene una fórmula de cesión."""
    for p in PATRONES_CAMBIO:
        m = re.search(p, texto, re.IGNORECASE)
        if m:
            after = texto[m.end():]
            boundary = re.search(r'[.!?]\s+', after)
            if boundary:
                split_at = m.end() + boundary.end()
                parte1 = texto[:split_at].strip()
                parte2 = texto[split_at:].strip()
                if parte2:
                    return parte1, parte2
    return texto, ""

def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_embed(chunk, encoder):
    if len(chunk) < 8000:  # mínimo 0.5s a 16kHz
        return None
    wav_tensor = torch.FloatTensor(chunk).unsqueeze(0)
    with torch.no_grad():
        embedding = encoder.encode_batch(wav_tensor)
    return embedding.squeeze().cpu().numpy()

def existe_en_diccionario(palabra):
    existe = dic.lookup(palabra.lower())
    print(f"'{palabra}': {'existe' if existe else 'no existe'}")
    return existe

def palabras_desconocidas(txt_path):
    texto = Path(txt_path).read_text(encoding="utf-8")
    texto = re.sub(r'\[\d+\.\d+s -> \d+\.\d+s\]', '', texto)
    texto = re.sub(r'^---$', '', texto, flags=re.MULTILINE)
    palabras = re.findall(r'\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+\b', texto)
    vistas = set()
    desconocidas = []
    for p in palabras:
        pl = p.lower()
        if pl not in vistas:
            vistas.add(pl)
            if not dic.lookup(pl) and not dic.lookup(pl.capitalize()):
                desconocidas.append(pl)
    output = Path(txt_path).with_stem(Path(txt_path).stem + "_desconocidas")
    output.write_text("\n".join(desconocidas), encoding="utf-8")
    print(f"{len(desconocidas)} palabras desconocidas guardadas en {output}")
    return desconocidas

indice_municipios = cargar_indice_municipios(Path.home() / "proyectos/transcripcion/dataset.json")

punct_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")

audio_path = Path.home() / "proyectos/transcripcion/tenerife1hora30.mp3"

print("Cargando audio...")
wav, _ = librosa.load(audio_path, sr=16000, mono=True)
print("Listo.")

print("Cargando modelo de embeddings de voz...")
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(Path.home() / "pretrained_models/spkrec-ecapa-voxceleb"),
    run_opts={"device": "cuda"}
)
print("Listo.")

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")

initial_prompt = (
    "con objeto de, en virtud de, a efectos de, "
    "el Cabildo Insular, el Consejo de Gobierno, "
    "el consejero, la consejera, el presidente, la presidenta, "
    "el Pleno, el orden del día, el expediente, "
    "Tenerife, Gran Canaria, Fuerteventura, Lanzarote, Mogán, Arucas, Gáldar"
)

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

output_path = audio_path.with_suffix(".txt")
duration = info.duration

with open(output_path, "w", encoding="utf-8") as f:
    speaker_start = 0.0
    prev_end = None
    primer_segmento = True
    ultimo_fue_cambio = False
    proximo_es_nuevo_orador = False  # "tiene la palabra" al final del segmento anterior
    for segment in segments:
        if segment.no_speech_prob > 0.6:
            continue
        if segment.compression_ratio > 2.4:
            continue

        if not primer_segmento and prev_end is not None:
            silencio = segment.start - prev_end
            if silencio >= SILENCIO_MINIMO:
                pre_start = max(speaker_start, prev_end - MAX_PRE_VENTANA)
                pre = wav[int(pre_start * 16000):int(prev_end * 16000)]
                post = wav[int(segment.start * 16000):int((segment.start + POST_VENTANA) * 16000)]
                duracion_pre = prev_end - pre_start
                print(f"  silencio={silencio:.2f}s pre={duracion_pre:.1f}s", end="")
                if duracion_pre >= MIN_PRE_DURACION and len(post) >= int(1.0 * 16000):
                    embed_pre = compute_embed(pre, encoder)
                    embed_post = compute_embed(post, encoder)
                    if embed_pre is not None and embed_post is not None:
                        dist = cosine_dist(embed_pre, embed_post)
                        print(f" dist={dist:.4f}", end="")
                        if dist > CAMBIO_THRESHOLD and not ultimo_fue_cambio:
                            print(" [CAMBIO audio]", end="")
                            f.write("\n---\n")
                            speaker_start = segment.start
                            ultimo_fue_cambio = True
                print()

        prev_end = segment.end

        texto = capitalizar_oraciones(punct_model.restore_punctuation(segment.text))

        # Cambio diferido del segmento anterior ("tiene la palabra" al final)
        if proximo_es_nuevo_orador and not ultimo_fue_cambio:
            f.write("\n---\n")
            speaker_start = segment.start
            ultimo_fue_cambio = True
        proximo_es_nuevo_orador = False

        if not primer_segmento and es_cambio_de_orador(texto):
            parte1, parte2 = split_en_cambio(texto)
            if parte2:
                # Patrón en el medio: partir siempre
                print(f"  [CAMBIO split] {parte1[-40:]!r} | {parte2[:40]!r}")
                f.write(parte1)
                f.write('\n' if parte1.rstrip().endswith(('.', '?', '!')) else ' ')
                f.write("\n---\n")
                speaker_start = segment.start
                ultimo_fue_cambio = True
                texto = parte2
            else:
                # Patrón al final: diferir al segmento siguiente
                print(f"  [tiene la palabra → deferido]")
                proximo_es_nuevo_orador = True

        print(f"[{segment.end/duration*100:.1f}%] {texto}")
        f.write(texto)
        if texto.rstrip().endswith(('.', '?', '!')):
            f.write('\n')
        else:
            f.write(' ')
        f.flush()
        primer_segmento = False
        ultimo_fue_cambio = False

desconocidas = palabras_desconocidas(output_path)
sugerir_correcciones(output_path, desconocidas, indice_municipios, threshold=50)
aplicar_correcciones(output_path, desconocidas, indice_municipios, threshold=81)
