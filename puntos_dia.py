#!/usr/bin/env python3
import sys
import json
import re
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
NUM_CTX = 8192
CONTEXT_CHARS = 1200

TRIGGER_RE = re.compile(
    r'(orden del d[ií]a'
    r'|punto [uú]nico'
    r'|punto n[uú]mero'
    r'|punto \d+'
    r'|\bprimer punto\b'
    r'|\bsiguiente punto\b'
    r'|pasamos al punto'
    r'|se pasa al punto'
    r'|punto del orden)',
    re.IGNORECASE,
)

PROMPT = """Dado el siguiente fragmento de un acta de pleno, extrae TODOS los puntos del orden del día que aparezcan. Solo los que sean obvios, no los inventes e ignora alusiones a plenos anteriores.

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:
{"puntos": [
  {
    "numero": número de orden o null si no se especifica,
    "titulo": "descripción breve del asunto tratado",
    "resumen": "resumen de una o dos frases de lo que se trató o debatió",
    "texto_original": "frase literal donde se anuncia o introduce el punto"
  }
]}

Si no hay puntos del orden del día devuelve: {"puntos": []}

FRAGMENTO:
"""


def extract_windows(text: str) -> list[tuple[int, str]]:
    windows = []
    last_end = -1
    for m in TRIGGER_RE.finditer(text):
        start = max(0, m.start() - CONTEXT_CHARS)
        end = min(len(text), m.end() + CONTEXT_CHARS)
        if start < last_end:
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
        return json.loads(match.group()).get("puntos", [])
    except json.JSONDecodeError:
        return []


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 puntos_dia.py acta.txt")
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
        print("No se detectaron menciones al orden del día en el texto.")
        out = path.with_suffix(".puntos.json")
        out.write_text(
            json.dumps({"archivo": path.name, "total": 0, "puntos": []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Guardado en: {out}")
        return

    todos = []
    seen = set()
    for i, (_, fragment) in enumerate(windows, 1):
        print(f"  [{i}/{len(windows)}] {len(fragment)} chars...", end=" ", flush=True)
        encontrados = call_ollama(fragment)
        nuevos = 0
        for p in encontrados:
            key = (p.get("texto_original") or "")[:80]
            if key and key not in seen:
                seen.add(key)
                todos.append(p)
                nuevos += 1
        print(f"{nuevos} punto(s)" if nuevos else "sin resultado")

    out = path.with_suffix(".puntos.json")
    out.write_text(
        json.dumps({"archivo": path.name, "total": len(todos), "puntos": todos},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'─' * 60}")
    print(f"TOTAL PUNTOS: {len(todos)}")
    print(f"{'─' * 60}")
    for i, p in enumerate(todos, 1):
        numero = p.get("numero")
        titulo = p.get("titulo") or ""
        resumen = p.get("resumen") or ""
        etiqueta = f"Punto {numero}" if numero else f"Punto {i}"
        print(f"\n  {etiqueta}. {titulo}")
        if resumen:
            print(f"     {resumen}")

    print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()
