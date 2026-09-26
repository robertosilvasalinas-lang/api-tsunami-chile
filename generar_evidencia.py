"""
generar_evidencia.py — Llama a la API (que debe estar encendida) y guarda la
transcripción de las llamadas exigidas en docs/evidencia_llamadas.md.

Uso (con la API corriendo en otra terminal):
    python generar_evidencia.py                      # usa http://localhost:8000
    python generar_evidencia.py https://mi-api.onrender.com
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
SALIDA = Path(__file__).resolve().parent / "docs" / "evidencia_llamadas.md"

LLAMADAS = [
    ("Estado del servicio", "GET", "/health", None),
    ("Metadatos del modelo", "GET", "/model-info", None),
    ("Predicción exitosa", "POST", "/predict",
     {"magnitude": 7.8, "depth": 20.0, "lat": -22.5, "lon": -70.9, "mag_type": "mww"}),
    ("Predicción por lote", "POST", "/predict-batch", {"sismos": [
        {"magnitude": 7.8, "depth": 20.0, "lat": -22.5, "lon": -70.9, "mag_type": "mww"},
        {"magnitude": 4.6, "depth": 110.0, "lat": -30.1, "lon": -69.5, "mag_type": "mb"}]}),
    ("Entrada inválida (falta 'lon' y magnitud fuera de rango) → 422", "POST", "/predict",
     {"magnitude": 12.5, "depth": 20.0, "lat": -30.0}),
]

# Las instancias gratuitas se duermen: el primer arranque puede tardar más de un minuto.
print(f"Despertando el servicio en {BASE} (puede tardar hasta 2 minutos)...")
for intento in range(1, 4):
    try:
        requests.get(BASE + "/health", timeout=120)
        print("Servicio despierto.\n")
        break
    except requests.RequestException as exc:
        print(f"  intento {intento}/3 sin respuesta ({type(exc).__name__}); reintentando...")
else:
    raise SystemExit(
        f"El servicio en {BASE} no respondió. Revisa en Render que el estado sea 'Live' "
        "y mira los logs del servicio."
    )

lineas = [f"# Evidencia de llamadas a la API\n\nServidor: `{BASE}` — generado el {datetime.now():%Y-%m-%d %H:%M}\n"]
for titulo, metodo, ruta, cuerpo in LLAMADAS:
    try:
        r = requests.request(metodo, BASE + ruta, json=cuerpo, timeout=120)
    except requests.ConnectionError:
        raise SystemExit(f"No hay respuesta en {BASE}. ¿Está encendida la API (uvicorn)?")
    print(f"{r.status_code}  {metodo} {ruta}")
    lineas.append(f"## {titulo}\n\n`{metodo} {ruta}` → **HTTP {r.status_code}**\n")
    if cuerpo is not None:
        lineas.append("Petición:\n```json\n" + json.dumps(cuerpo, indent=2, ensure_ascii=False) + "\n```\n")
    lineas.append("Respuesta:\n```json\n" + json.dumps(r.json(), indent=2, ensure_ascii=False) + "\n```\n")

SALIDA.parent.mkdir(exist_ok=True)
SALIDA.write_text("\n".join(lineas), encoding="utf-8")
print(f"\nEvidencia guardada en {SALIDA}")
