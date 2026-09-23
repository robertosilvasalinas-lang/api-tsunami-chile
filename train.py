"""
train.py — Entrena el clasificador de alerta de tsunami y lo serializa.

Qué hace, en orden:
  1. Descarga el catálogo sísmico público de USGS para Chile (o usa la copia local).
  2. Limpia los datos y muestra una exploración básica (EDA).
  3. Separa entrenamiento / prueba con semilla fija (estratificado).
  4. Entrena un Pipeline COMPLETO (preprocesamiento + RandomForest).
  5. Evalúa con validación cruzada y en el conjunto de prueba.
  6. Guarda model/model.pkl y model/metadata.json.

Uso:
    python train.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuración (rutas RELATIVAS al proyecto: funcionan en cualquier computador)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sismos_chile_usgs.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

SEED = 42
MODEL_VERSION = "1.0.0"

# Catálogo público de USGS (Servicio Geológico de EE.UU.). Ventana de tiempo FIJA
# para que cualquiera que ejecute este script obtenga los mismos datos.
# Ventana desde 2010: verificamos que USGS no registra el flag de tsunami en eventos
# anteriores (quedaban en 0 aunque hubiera sismos mayores), así que incluir esos años
# introducía etiquetas falsamente negativas. Magnitud >= 5.0 para no inundar el
# dataset de eventos pequeños que nunca generan alerta.
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_PARAMS = {
    "format": "geojson",
    "starttime": "2010-01-01",
    "endtime": "2026-01-01",
    "minlatitude": -56,
    "maxlatitude": -17,
    "minlongitude": -80,
    "maxlongitude": -66,
    "minmagnitude": 5.0,
    "orderby": "time-asc",
    "limit": 20000,
}

NUM_COLS = ["magnitude", "depth", "lat", "lon"]
CAT_COLS = ["mag_type"]
FEATURES = NUM_COLS + CAT_COLS  # ORDEN que la API debe respetar
TARGET = "tsunami"


# ---------------------------------------------------------------------------
# 1. Obtención de datos
# ---------------------------------------------------------------------------
def descargar_datos() -> pd.DataFrame:
    """Devuelve el dataset. Usa la copia local si existe; si no, la descarga de USGS."""
    if DATA_PATH.exists():
        print(f"Usando copia local de los datos: {DATA_PATH.relative_to(ROOT)}")
        return pd.read_csv(DATA_PATH)

    print("Descargando catálogo sísmico desde USGS (puede tardar ~30 segundos)...")
    try:
        resp = requests.get(USGS_URL, params=USGS_PARAMS, timeout=120)
        resp.raise_for_status()
        eventos = resp.json()["features"]
    except Exception as exc:  # noqa: BLE001
        # Sin datos reales NO entrenamos: un modelo con datos inventados
        # no es defendible ni reproducible.
        sys.exit(
            f"\nERROR: no se pudo descargar el catálogo de USGS ({exc}).\n"
            "Revisa tu conexión a internet y vuelve a ejecutar: python train.py"
        )

    filas = []
    for ev in eventos:
        p = ev["properties"]
        lon, lat, depth = ev["geometry"]["coordinates"][:3]
        filas.append(
            {
                "event_id": ev["id"],
                "time": pd.to_datetime(p["time"], unit="ms", utc=True),
                "place": p.get("place"),
                "magnitude": p.get("mag"),
                "mag_type": p.get("magType"),
                "depth": depth,
                "lat": lat,
                "lon": lon,
                "tsunami": p.get("tsunami"),
            }
        )
    df = pd.DataFrame(filas)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Descargados {len(df)} sismos. Copia guardada en {DATA_PATH.relative_to(ROOT)}")
    return df


# ---------------------------------------------------------------------------
# 2. Limpieza y exploración
# ---------------------------------------------------------------------------
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="event_id").copy()
    for col in NUM_COLS + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # La variable objetivo y la magnitud son imprescindibles: sin ellas la fila se descarta.
    # Los nulos en otras columnas los resuelve el Pipeline (SimpleImputer).
    df = df.dropna(subset=["magnitude", TARGET])
    df[TARGET] = df[TARGET].astype(int)
    # Profundidades levemente negativas (sobre el nivel del mar) se llevan a 0
    df["depth"] = df["depth"].clip(lower=0)
    return df


def explorar(df: pd.DataFrame) -> None:
    print("\n================ EXPLORACIÓN DE DATOS ================")
    print(f"Filas: {len(df)} | Columnas predictoras: {FEATURES}")
    print("\nValores nulos por columna:")
    print(df[FEATURES + [TARGET]].isna().sum().to_string())
    print("\nBalance de clases (tsunami):")
    print(df[TARGET].value_counts().rename({0: "0 = sin flag", 1: "1 = flag tsunami"}).to_string())
    print(f"Proporción de positivos: {df[TARGET].mean():.2%}")
    if "time" in df.columns:
        periodo = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.year // 5 * 5
        tabla = df.assign(periodo=periodo).groupby("periodo")[TARGET].agg(["size", "sum"])
        tabla.columns = ["sismos", "con_alerta"]
        print("\nSismos y casos con alerta por quinquenio (para detectar años sin registro del flag):")
        print(tabla.to_string())

    print("\nTipos de magnitud (variable categórica):")
    print(df["mag_type"].value_counts(dropna=False).head(10).to_string())
    print("\nResumen numérico:")
    print(df[NUM_COLS].describe().round(2).to_string())
    print("=======================================================\n")


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
def construir_pipeline() -> Pipeline:
    numericas = Pipeline(
        [("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]
    )
    categoricas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
            # Categorías raras (<10 casos) o nunca vistas se agrupan como "infrecuentes"
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
        ]
    )
    pre = ColumnTransformer([("num", numericas, NUM_COLS), ("cat", categoricas, CAT_COLS)])
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",  # compensa que los positivos son muy pocos
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def regla_simple(X: pd.DataFrame) -> np.ndarray:
    """Línea base: 'alerta si magnitud >= 6.5 y profundidad <= 45 km'."""
    return ((X["magnitude"] >= 6.5) & (X["depth"] <= 45)).astype(int).to_numpy()


def metricas(y_true, y_pred, y_score=None) -> dict:
    m = {
        "precision_alerta": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_alerta": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_alerta": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    if y_score is not None:
        m["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        m["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return m


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    df = limpiar(descargar_datos())
    explorar(df)

    if len(df) < 500:
        print("ADVERTENCIA: el dataset tiene menos de 500 filas (mínimo de la tarea).")
    n_pos = int(df[TARGET].sum())
    if n_pos < 30:
        print(f"AVISO: solo hay {n_pos} casos con alerta; las métricas tendrán mucha varianza.")
    if n_pos < 10:
        sys.exit(f"ERROR: solo hay {n_pos} casos positivos; no es posible entrenar con rigor.")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(
        f"Entrenamiento: {len(X_train)} filas ({int(y_train.sum())} con alerta) | "
        f"Prueba: {len(X_test)} filas ({int(y_test.sum())} con alerta)"
    )
    if int(y_test.sum()) < 15:
        print(
            "AVISO: hay pocos casos con alerta en la prueba, así que esas métricas son\n"
            "       inestables. La validación cruzada (5 folds sobre todo el entrenamiento)\n"
            "       es la referencia más confiable para interpretar los resultados."
        )

    pipe = construir_pipeline()

    # Validación cruzada SOLO sobre entrenamiento (el test queda intacto → sin fuga)
    # Repetimos la validación cruzada 5 veces con particiones distintas: con tan pocos
    # casos positivos, una sola pasada da resultados que dependen mucho del azar.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1", "average_precision", "recall"])
    cv_metricas = {
        "cv_f1_alerta_media": round(float(cv_res["test_f1"].mean()), 4),
        "cv_f1_alerta_std": round(float(cv_res["test_f1"].std()), 4),
        "cv_pr_auc_media": round(float(cv_res["test_average_precision"].mean()), 4),
        "cv_recall_alerta_media": round(float(cv_res["test_recall"].mean()), 4),
    }
    print("\nValidación cruzada (5 folds x 5 repeticiones, entrenamiento):", cv_metricas)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(1)]

    test_metricas = metricas(y_test, y_pred, y_score)
    base_metricas = metricas(y_test, regla_simple(X_test))

    print("\n--- Reporte en conjunto de PRUEBA ---")
    print(classification_report(y_test, y_pred, target_names=["sin alerta", "alerta"], zero_division=0))
    print("Matriz de confusión [[VN, FP], [FN, VP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nModelo       :", test_metricas)
    print("Regla simple :", base_metricas)

    # ---------------- Serialización ----------------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)

    categorias = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"].categories_[0]
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estimator": type(pipe.named_steps["clf"]).__name__,
        "python_version": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "features": FEATURES,
        "numeric_features": NUM_COLS,
        "categorical_features": CAT_COLS,
        "mag_type_categories_seen": [str(c) for c in categorias],
        "target": {
            "name": TARGET,
            "descripcion": "Flag 'tsunami' de USGS: 1 = evento grande en región oceánica "
            "con información del sistema de alerta de tsunamis de NOAA.",
            "clases": {"0": "SIN ALERTA", "1": "ALERTA"},
        },
        "dataset": {
            "fuente": "USGS Earthquake Catalog (ComCat), API FDSN Event",
            "url": USGS_URL,
            "parametros": USGS_PARAMS,
            "n_filas": int(len(df)),
            "n_positivos": n_pos,
            "tasa_positivos": round(float(df[TARGET].mean()), 4),
        },
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "metric": "f1_alerta",
        "value": test_metricas["f1_alerta"],
        "metrics_cv": cv_metricas,
        "metrics_test": test_metricas,
        "baseline_regla_simple_test": base_metricas,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    tam_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"\nModelo guardado en {MODEL_PATH.relative_to(ROOT)} ({tam_mb:.2f} MB)")
    print(f"Metadatos guardados en {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
