#!/usr/bin/env python3
"""
Corrige errores de transcripción de Whisper en un fichero de texto en español.
Usa la API de Claude para corregir errores fonéticos propios del acento canario
y otras confusiones habituales de Whisper.
"""

import sys
from pathlib import Path
import anthropic

MODELO = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
Eres un corrector de transcripciones automáticas de discursos políticos formales en español.
La transcripción fue generada con Whisper a partir del audio de un pleno de algún sitio de España, puede ser Madrid o Canarias entre otros.

Haz exactamente estas tres cosas, nada más:

1. ERRORES ORTOGRÁFICOS EVIDENTES: corrige solo palabras que estén claramente mal escritas
   o que no tengan sentido en el contexto (errores fonéticos de Whisper). En caso de duda,
   deja el texto original intacto, corrige solo las casos evidentes.

2. PUNTUACIÓN: revisa puntos y comas. Whisper a veces los pone mal o los omite. Corrígelos
   si es evidente, pero no reescribas frases.

3. LÍNEAS PARTIDAS: si una línea no termina en punto, interrogación o exclamación,
   únela con la siguiente separándola con un espacio. Whisper corta el audio por
   segmentos, no por oraciones, y esos cortes no deben aparecer en el texto final.


Prohibido: no mejores el estilo, no reformules, no añadas ni elimines ideas o cifras.
Conserva los marcadores [CAMBIO DE ORADOR] exactamente como aparecen.
Devuelve únicamente el texto corregido, sin explicaciones ni comentarios.
Muchas gracias por tu ayuda.
"""


def corregir_seccion(client: anthropic.Anthropic, seccion: str, numero: int, total: int) -> str:
    print(f"\n[{numero}/{total}] Corrigiendo sección ({len(seccion):,} chars)...", flush=True)

    resultado = []
    with client.messages.stream(
        model=MODELO,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Corrige los errores de transcripción del siguiente fragmento. "
                    "Devuelve únicamente el texto corregido, sin añadir nada más:\n\n"
                    + seccion
                ),
            }
        ],
    ) as stream:
        for evento in stream.text_stream:
            print(evento, end="", flush=True)
            resultado.append(evento)

    print()
    return partir_frases_largas("".join(resultado))


def partir_frases_largas(texto: str, max_palabras: int = 50) -> str:
    # Primero unir líneas que no terminan en fin de oración
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

    # Luego partir en el siguiente punto tras max_palabras
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
    print(f"Dividido en {total} secciones por [CAMBIO DE ORADOR].")

    client = anthropic.Anthropic()
    secciones_corregidas = []
    for i, seccion in enumerate(secciones, start=1):
        if seccion.strip():
            corregida = corregir_seccion(client, seccion.strip(), i, total)
            secciones_corregidas.append(corregida)
        else:
            secciones_corregidas.append(seccion)

    corregido = "\n\n[CAMBIO DE ORADOR]\n\n".join(secciones_corregidas)

    SALIDA.write_text(corregido, encoding="utf-8")
    print(f"\nFichero guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
