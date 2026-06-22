#!/usr/bin/env python3
import sys
import json
import re
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
NUM_CTX = 4096
CONTEXT_CHARS = 800  # chars antes y después de la frase detectada

# Frases que indican una votación activa en la sesión
TRIGGER_RE = re.compile(
    r'(comienza la votaci[oó]n'
    r'|se aprueba'
    r'|se rechaza'
    r'|queda aprobad'
    r'|queda rechazad'
    r'|votos? a favor'
    r'|votos? en contra'
    r'|\babstenciones?\b'
    r'|aprobado por unanimidad'
    r'|por unanimidad)',
    re.IGNORECASE,
)

PROMPT = """Dado el siguiente fragmento de un acta de pleno, extrae TODAS las votaciones que contenga.

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:
{"votaciones": [
  {
    "punto": "descripción del asunto votado (del contexto previo al anuncio)",
    "resultado": "aprobado" | "rechazado" | "retirado" | "aplazado" | "otro",
    "unanimidad": true | false,
    "favor": número o null,
    "contra": número o null,
    "abstenciones": número o null,
    "texto_original": "frase literal donde se anuncia el resultado"
  }
]}

Si no hay votaciones reales devuelve: {"votaciones": []}

FRAGMENTO:
"""


def extract_windows(text: str) -> list[tuple[int, str]]:
    """Devuelve ventanas de contexto alrededor de cada trigger, sin solapar."""
    windows = []
    last_end = -1
    for m in TRIGGER_RE.finditer(text):
        start = max(0, m.start() - CONTEXT_CHARS)
        end = min(len(text), m.end() + CONTEXT_CHARS)
        if start < last_end:
            # Ampliar la ventana anterior en lugar de crear una nueva
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
        last_end = end
    return [(s, text[s:e]) for s, e in windows]


def call_ollama(fragment: str) -> list[dict]:
    payload = {
        "model": MODEL,
        "prompt": PROMPT + fragment,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return []
    try:
        return json.loads(match.group()).get("votaciones", [])
    except json.JSONDecodeError:
        return []


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 votaciones.py acta.txt")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: no existe {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    windows = extract_windows(text)

    print(f"Archivo: {path.name} ({len(text):,} chars)")
    print(f"Ventanas con trigger: {len(windows)} | Modelo: {MODEL}\n")

    if not windows:
        print("No se detectaron frases de votación en el texto.")
        out = path.with_suffix(".votaciones.json")
        out.write_text(
            json.dumps({"archivo": path.name, "total": 0, "votaciones": []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Guardado en: {out}")
        return

    todas = []
    seen = set()
    for i, (_, fragment) in enumerate(windows, 1):
        print(f"  [{i}/{len(windows)}] {len(fragment)} chars...", end=" ", flush=True)
        encontradas = call_ollama(fragment)
        nuevas = 0
        for v in encontradas:
            key = (v.get("texto_original") or "")[:80]
            if key and key not in seen:
                seen.add(key)
                todas.append(v)
                nuevas += 1
        print(f"{nuevas} votación(es)" if nuevas else "sin resultado")

    out = path.with_suffix(".votaciones.json")
    out.write_text(
        json.dumps({"archivo": path.name, "total": len(todas), "votaciones": todas},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'─' * 60}")
    print(f"TOTAL VOTACIONES: {len(todas)}")
    print(f"{'─' * 60}")
    for i, v in enumerate(todas, 1):
        etiqueta = v.get("resultado", "?").upper()
        if v.get("unanimidad"):
            etiqueta += " [UNANIMIDAD]"
        favor = v.get("favor")
        contra = v.get("contra")
        abst = v.get("abstenciones")
        if favor is not None or contra is not None:
            etiqueta += f"  {favor}F / {contra}C / {abst}A"
        punto = v.get("punto") or ""
        texto = (v.get("texto_original") or "")[:80]
        print(f"\n  {i}. {etiqueta}")
        if punto:
            print(f"     Punto: {punto}")
        print(f"     \"{texto}...\"")

    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()
