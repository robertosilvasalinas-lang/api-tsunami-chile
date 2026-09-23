"""
predict.py — Verifica que model/model.pkl carga en un proceso DISTINTO al que lo
entrenó y clasifica tres sismos de prueba.  Uso:  python predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

print(f"Cargando el modelo desde '{MODEL_PATH.name}'...")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"No se pudo cargar el modelo. ¿Ejecutaste 'python train.py'? Error: {e}")
print("¡Modelo cargado exitosamente!\n")

nuevos_sismos = pd.DataFrame([
    {"descripcion": "Sismo 1: Leve y profundo", "magnitude": 4.6, "depth": 110.0, "lat": -30.1, "lon": -69.5, "mag_type": "mb"},
    {"descripcion": "Sismo 2: Moderado en la costa", "magnitude": 5.8, "depth": 35.0, "lat": -36.8, "lon": -73.0, "mag_type": "mww"},
    {"descripcion": "Sismo 3: Fuerte, superficial y en el mar", "magnitude": 7.8, "depth": 20.0, "lat": -22.5, "lon": -70.9, "mag_type": "mww"},
])

X_nuevos = nuevos_sismos[["magnitude", "depth", "lat", "lon", "mag_type"]]
predicciones = model.predict(X_nuevos)
probabilidades = model.predict_proba(X_nuevos)
clases = list(model.classes_)

print("--- RESULTADOS DE CLASIFICACIÓN ---")
for i, row in nuevos_sismos.iterrows():
    pred = int(predicciones[i])
    prob = probabilidades[i][clases.index(pred)] * 100
    estado = "🚨 ALERTA DE TSUNAMI" if pred == 1 else "✅ Sin alerta"
    print(f"\n{row['descripcion']}")
    print(f"  - Magnitud: {row['magnitude']} | Profundidad: {row['depth']} km")
    print(f"  {estado} (confianza del modelo: {prob:.2f}%)")
