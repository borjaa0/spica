#!/usr/bin/env python3
import sys
import json
import re
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
NUM_CTX = 4096
CONTEXT_CHARS = 1600  # chars antes y después de la frase detectada

# Frases que indican una votación activa en la sesión
TRIGGER_RE = re.compile(
    r'(comienza la votaci[oó]n'
    r'|comienza la votaci[oó]n electrónica'
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

PROMPT = """Dado el siguiente fragmento de un acta de pleno, extrae TODAS las votaciones que contenga. Ignora cualquier mención a votaciones que ocurrieron en comisión plenaria, en junta de portavoces o en reuniones anteriores — solo las votaciones del pleno actual.

REGLA 1 — campo "punto":
- Si el texto dice "punto número X", "el punto X", "punto único" → SIEMPRE es un punto del orden del día: "Punto X - [descripción breve]". Da igual que el contenido mencione una "propuesta relativa a..." o un ayuntamiento beneficiario: si lo introduce como "punto número X" es un Punto, no una Propuesta.
- Si es una propuesta de resolución presentada por un grupo político (Grupo Mixto, Grupo Popular, Grupo Socialista, Coalición Canaria, etc.) → "Propuesta X - [Grupo]".
- Si un punto tiene enmiendas y luego dictamen, crea dos entradas: "Punto X - enmiendas [grupo]" y "Punto X - dictamen".

REGLA 2 — campos numéricos (convierte palabras a dígitos):
- "favor": SOLO votos a favor. "uno a favor", "dieciocho votos a favor" → favor: 1 o 18.
- "contra": SOLO votos en contra. "diez en contra", "diecinueve votos en contra" → contra: 10 o 19.
- "abstenciones": SOLO abstenciones. "once abstenciones", "diez abstenciones" → abstenciones: 11 o 10.
- NUNCA pongas abstenciones en el campo "contra" ni votos a favor en "abstenciones".
- Pon null SOLO si ese dato no aparece en el texto.

EJEMPLOS:

Texto: "Comienza la votación electrónica, punto 11, dictamen de la Comisión Plenaria. El punto número 12. Falta proclamar. Sí, se aprueba por unanimidad"
→ {"votaciones": [{"punto": "Punto 11 - dictamen comision plenaria", "resultado": "aprobado", "unanimidad": true, "favor": null, "contra": null, "abstenciones": null, "texto_original": "Se aprueba por unanimidad."}]}

Texto: "El punto número once es la propuesta relativa a la cesión del inmueble al Ayuntamiento de Güímar. Comienza la votación electrónica. Se aprueba por unanimidad."
→ {"votaciones": [{"punto": "Punto 11 - cesión inmueble Ayuntamiento Güímar", "resultado": "aprobado", "unanimidad": true, "favor": null, "contra": null, "abstenciones": null, "texto_original": "Se aprueba por unanimidad."}]}

Texto: "Se aprueba por mayoría, 18 votos a favor y 11 abstenciones."
→ favor: 18, contra: null, abstenciones: 11

Texto: "Se rechaza por mayoría, veintisiete votos en contra y uno a favor."
→ favor: 1, contra: 27, abstenciones: null   ← "uno a favor" es favor: 1, aunque sea un solo voto

Texto: "Se aprueba por mayoría, dieciocho votos a favor y diez en contra."
→ favor: 18, contra: 10, abstenciones: null   ← "en contra" va en contra, NO en abstenciones

Texto: "Se aprueba por mayoría, 27 votos a favor y 1 en contra."
→ favor: 27, contra: 1, abstenciones: null   ← aunque sea solo 1, "en contra" va en contra

Texto: "El punto número 12, propuesta relativa a la aprobación inicial del expediente de modificación de los créditos número 4 del presupuesto 2026."
→ {"votaciones": []} — esto es solo la presentación del punto, no hay resultado de votación.


Devuelve ÚNICAMENTE un objeto JSON con esta estructura:
{"votaciones": [
  {
    "punto": "descripción precisa según las reglas anteriores",
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


_NUMEROS_ES = {
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciséis": 16, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidós": 22,
    "veintidos": 22, "veintitrés": 23, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiséis": 26, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30,
}
_NUM_PAT = r'(\d+|' + '|'.join(_NUMEROS_ES.keys()) + r')'
_FAVOR_RE = re.compile(_NUM_PAT + r'\s*(?:votos?\s+)?a favor', re.IGNORECASE)
_CONTRA_RE = re.compile(_NUM_PAT + r'\s*(?:votos?\s+)?en contra', re.IGNORECASE)
_ABST_RE = re.compile(_NUM_PAT + r'\s*abstenciones?', re.IGNORECASE)
_VOTACION_PUNTO_RE = re.compile(
    r'comienza la votaci[oó]n electr[oó]nica,?\s+punto\s+(\d+)',
    re.IGNORECASE,
)


def _parse_num(s: str) -> int:
    s = s.strip().lower()
    return int(s) if s.isdigit() else _NUMEROS_ES.get(s, 0)


def corregir_numeros(v: dict) -> dict:
    texto = v.get("texto_original", "")
    m = _FAVOR_RE.search(texto)
    v["favor"] = _parse_num(m.group(1)) if m else None
    m = _CONTRA_RE.search(texto)
    v["contra"] = _parse_num(m.group(1)) if m else None
    m = _ABST_RE.search(texto)
    v["abstenciones"] = _parse_num(m.group(1)) if m else None
    return v


def corregir_punto(v: dict, fragment: str) -> dict:
    m = _VOTACION_PUNTO_RE.search(fragment)
    if not m:
        return v
    numero_correcto = m.group(1)
    # Solo corregir si el texto_original aparece DESPUÉS del trigger en el fragmento
    texto = v.get("texto_original", "")
    pos_texto = fragment.find(texto[:60]) if texto else -1
    if pos_texto == -1 or pos_texto < m.end():
        return v
    punto = v.get("punto", "")
    corregido = re.sub(r'(?i)^Punto\s+\d+', f'Punto {numero_correcto}', punto)
    if corregido != punto:
        v["punto"] = corregido
    return v


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
        votaciones = json.loads(match.group()).get("votaciones", [])
    except json.JSONDecodeError:
        return []
    return [corregir_numeros(corregir_punto(v, fragment)) for v in votaciones]


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
            key = (v.get("punto") or "") + "|" + (v.get("texto_original") or "")[:80]
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
