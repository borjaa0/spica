#!/usr/bin/env python3
"""
Corrige errores de transcripción de Whisper en un fichero de texto en español.
Usa Ollama (qwen2.5:7b) para corregir errores fonéticos y de puntuación.
"""

import sys
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO = "qwen2.5:3b"
NUM_CTX = 4096
MAX_WORKERS = 2

PROMPT = """\
Eres un corrector de transcripciones automáticas de discursos políticos formales en español.
La transcripción fue generada con Whisper a partir del audio de un pleno de algún sitio de España, puede ser Madrid o Canarias entre otros.

Haz exactamente estas tres cosas, nada más:

1. ERRORES ORTOGRÁFICOS EVIDENTES: corrige solo palabras que estén claramente mal escritas
   o que no tengan sentido en el contexto (errores fonéticos de Whisper). En caso de duda,
   deja el texto original intacto, corrige solo los casos evidentes.

2. PUNTUACIÓN: revisa puntos y comas. Whisper a veces los pone mal o los omite. Corrígelos
   si es evidente, pero no reescribas frases.

3. LÍNEAS PARTIDAS: si una línea no termina en punto, interrogación o exclamación,
   únela con la siguiente separándola con un espacio. Whisper corta el audio por
   segmentos, no por oraciones, y esos cortes no deben aparecer en el texto final.

Prohibido: no mejores el estilo, no reformules, no añadas ni elimines ideas o cifras.
Devuelve únicamente el texto corregido, sin explicaciones ni comentarios.

TEXTO:
"""


def corregir_seccion(seccion: str, numero: int, total: int) -> str:
    print(f"\n[{numero}/{total}] Corrigiendo sección ({len(seccion):,} chars)...", flush=True)

    payload = {
        "model": MODELO,
        "prompt": PROMPT + seccion,
        "stream": True,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
    resp.raise_for_status()

    resultado = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        print(token, end="", flush=True)
        resultado.append(token)
        if data.get("done"):
            break

    print()
    return partir_frases_largas("".join(resultado))


def partir_frases_largas(texto: str, max_palabras: int = 50) -> str:
    lineas = texto.splitlines()
    unidas = []
    for linea in lineas:
        stripped = linea.strip()
        if not stripped:
            unidas.append("")
            continue
        if unidas and unidas[-1] and not unidas[-1].rstrip().endswith((".", "?", "!", ":", "—")):
            unidas[-1] = unidas[-1].rstrip() + " " + stripped
        else:
            unidas.append(stripped)
    texto_unido = "\n".join(unidas)

    oraciones = texto_unido.split(". ")
    parrafos = []
    acumulado = []
    conteo = 0
    for oracion in oraciones:
        palabras = len(oracion.split())
        if conteo + palabras > max_palabras and acumulado:
            parrafos.append(". ".join(acumulado) + ".")
            acumulado = [oracion]
            conteo = palabras
        else:
            acumulado.append(oracion)
            conteo += palabras
    if acumulado:
        parrafos.append(". ".join(acumulado))
    return "\n".join(parrafos)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 corregir_transcripcion.py <fichero.txt>", file=sys.stderr)
        sys.exit(1)

    ENTRADA = Path(sys.argv[1])
    SALIDA = ENTRADA.with_stem(ENTRADA.stem + "_corregido")

    if not ENTRADA.exists():
        print(f"Error: no se encuentra el fichero {ENTRADA}", file=sys.stderr)
        sys.exit(1)

    print(f"Leyendo {ENTRADA}...")
    texto_original = ENTRADA.read_text(encoding="utf-8")

    secciones = texto_original.split("[CAMBIO DE ORADOR]")
    total = len(secciones)
    print(f"Dividido en {total} secciones | Modelo: {MODELO}")

    trabajos = {
        i: seccion for i, seccion in enumerate(secciones) if seccion.strip()
    }

    resultados = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {
            executor.submit(corregir_seccion, seccion, i + 1, total): i
            for i, seccion in trabajos.items()
        }
        for futuro in as_completed(futuros):
            i = futuros[futuro]
            resultados[i] = futuro.result()

    secciones_corregidas = [
        resultados.get(i, seccion) for i, seccion in enumerate(secciones)
    ]

    corregido = "\n\n[CAMBIO DE ORADOR]\n\n".join(secciones_corregidas)
    SALIDA.write_text(corregido, encoding="utf-8")
    print(f"\nFichero guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
