#!/usr/bin/env python3
"""
Corrige errores de transcripción de Whisper en un fichero de texto en español.
Usa la API de Claude para corregir errores fonéticos propios del acento canario
y otras confusiones habituales de Whisper.
"""

import sys
from pathlib import Path
import anthropic

ENTRADA = Path(__file__).parent / "tenerife3.txt"
SALIDA = Path(__file__).parent / "tenerife3_corregido.txt"
MODELO = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
Eres un corrector experto en transcripciones automáticas de discursos políticos formales en español.
La transcripción fue generada con Whisper a partir del audio de un pleno del Cabildo de Tenerife.
Los ponentes incluyen políticos canarios con diferentes acentos regionales.

Whisper comete errores fonéticos característicos: palabras técnicas, económicas o jurídicas
son sustituidas por palabras fonéticamente similares pero sin sentido en el contexto.
Tu tarea es detectar y corregir esos errores basándote ÚNICAMENTE en el contexto del discurso.

Reglas estrictas:
1. Corrige solo lo que claramente esté mal. No cambies lo que tenga sentido.
2. Mantén el registro formal y el estilo original del orador.
3. No añadas ni elimines ideas, cifras ni argumentos.
4. Conserva la estructura del texto tal como aparece, incluyendo los marcadores [CAMBIO DE ORADOR].
5. Devuelve únicamente el texto corregido, sin explicaciones ni comentarios.
"""


def corregir_con_claude(texto: str) -> str:
    client = anthropic.Anthropic()

    print(f"Enviando {len(texto):,} caracteres a {MODELO}...", flush=True)

    texto_corregido = []
    with client.messages.stream(
        model=MODELO,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Corrige los errores de transcripción del siguiente texto. "
                    "Devuelve el texto completo corregido manteniendo los marcadores "
                    "[CAMBIO DE ORADOR] exactamente como aparecen:\n\n"
                    + texto
                ),
            }
        ],
    ) as stream:
        for evento in stream.text_stream:
            print(evento, end="", flush=True)
            texto_corregido.append(evento)

    print()
    return "".join(texto_corregido)


def main():
    if not ENTRADA.exists():
        print(f"Error: no se encuentra el fichero {ENTRADA}", file=sys.stderr)
        sys.exit(1)

    print(f"Leyendo {ENTRADA}...")
    texto_original = ENTRADA.read_text(encoding="utf-8")

    print("Corrigiendo con Claude...")
    corregido = corregir_con_claude(texto_original)

    SALIDA.write_text(corregido, encoding="utf-8")
    print(f"\nFichero guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
