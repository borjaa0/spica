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

REGLAS:
- Un punto del orden del día es un asunto oficial de la sesión, anunciado como "el punto número X", "primer punto", "punto único", etc.
- El número debe extraerse del texto cuando aparezca. "punto número 14" → numero: 14. "primer punto" → numero: 1. No pongas null si el número está en el texto.
- Si varios puntos se debaten conjuntamente (ej: "se debatirá conjuntamente con el 13 y el 14"), crea una entrada por cada número mencionado con su número correcto.
- NO extraigas como punto separado: debates o explicaciones dentro de un punto, votaciones de enmiendas, encargos mencionados de pasada.
- El campo "titulo" debe derivarse ÚNICAMENTE del `texto_original` que estás incluyendo en la respuesta. No uses información de otras frases del fragmento para rellenar el titulo de un punto.

EJEMPLOS:

Texto: "comenzamos el orden del día del Pleno Ordinario con el primer punto que es la aprobación de las actas"
→ {"puntos": [{"numero": 1, "titulo": "Aprobación de las actas", ...}]}

Texto: "el punto número 12 se debatirá conjuntamente con el 13 y el 14"
→ {"puntos": [{"numero": 12, ...}, {"numero": 13, ...}, {"numero": 14, ...}]}  ← tres entradas separadas

Texto: "Procederíamos a la votación del punto número 14, que es una corrección de errores materiales en el anexo 2"
→ {"puntos": [{"numero": 14, "titulo": "Corrección de errores materiales en subvenciones nominativas", ...}]}

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:
{"puntos": [
  {
    "numero": número entero o null si no se especifica,
    "titulo": "descripción breve del asunto tratado",
    "resumen": "resumen de una o dos frases de lo que se trató o debatió",
    "texto_original": "frase literal donde se anuncia o introduce el punto"
  }
]}

Si no hay puntos del orden del día devuelve: {"puntos": []}

FRAGMENTO:
"""


_NUMEROS_ES_P = {
    "uno": 1, "primera": 1, "primero": 1, "dos": 2, "segundo": 2, "tres": 3,
    "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciséis": 16, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20,
}
_NUM_PAT_P = r'(\d+|' + '|'.join(_NUMEROS_ES_P.keys()) + r')'
_PUNTO_NUM_RE = re.compile(
    r'(?:punto\s+n[uú]mero\s+' + _NUM_PAT_P + r'|' +
    r'\bprimer\s+punto\b|' +
    r'punto\s+' + _NUM_PAT_P + r'\b)',
    re.IGNORECASE,
)


def corregir_numero_punto(p: dict) -> dict:
    if p.get("numero") is not None:
        return p
    texto = p.get("texto_original", "")
    m = _PUNTO_NUM_RE.search(texto)
    if not m:
        return p
    raw = m.group(1) or m.group(2)
    if raw is None:
        p["numero"] = 1  # "primer punto"
        return p
    raw = raw.strip().lower()
    p["numero"] = int(raw) if raw.isdigit() else _NUMEROS_ES_P.get(raw, None)
    return p


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
        puntos = json.loads(match.group()).get("puntos", [])
    except json.JSONDecodeError:
        return []
    return [corregir_numero_punto(p) for p in puntos]


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
            key = str(p.get("numero")) + "|" + (p.get("texto_original") or "")[:80]
            if key and key not in seen:
                seen.add(key)
                todos.append(p)
                nuevos += 1
        print(f"{nuevos} punto(s)" if nuevos else "sin resultado")

    _ANUNCIO_RE = re.compile(
        r'punto\s+n[uú]mero|punto\s+\d+|primer\s+punto|punto\s+[uú]nico|siguiente\s+punto|pasamos\s+al\s+punto',
        re.IGNORECASE,
    )
    todos = [
        p for p in todos
        if p.get("numero") is not None or _ANUNCIO_RE.search(p.get("texto_original", ""))
    ]

    def _mencion_directa(texto: str, numero: int) -> bool:
        """True si texto_original introduce el punto número como sujeto principal."""
        pat = re.compile(
            r'punto\s+n[uú]mero\s+' + str(numero) + r'\b'
            r'|punto\s+' + str(numero) + r'\b',
            re.IGNORECASE,
        )
        m = pat.search(texto)
        if not m:
            return False
        return m.start() < len(texto) * 0.6

    # Deduplicar por numero: preferir mención directa sobre incidental, luego más largo
    vistos = {}
    for p in todos:
        n = p.get("numero")
        if n is None:
            continue
        if n not in vistos:
            vistos[n] = p
        else:
            existing = vistos[n]
            new_direct = _mencion_directa(p.get("texto_original", ""), n)
            old_direct = _mencion_directa(existing.get("texto_original", ""), n)
            if new_direct and not old_direct:
                vistos[n] = p
            elif not new_direct and old_direct:
                pass
            elif len(p.get("texto_original", "")) > len(existing.get("texto_original", "")):
                vistos[n] = p
    sin_numero = [p for p in todos if p.get("numero") is None]
    todos = list(vistos.values()) + sin_numero
    todos.sort(key=lambda p: (p.get("numero") or 9999))

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
