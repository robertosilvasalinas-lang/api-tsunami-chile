"""
main.py — API de inferencia: alerta de tsunami para sismos en Chile.

Levantar en local:
    uvicorn app.main:app --reload --port 8000
Documentación interactiva:
    http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    InfoModelo,
    LoteSismos,
    Prediccion,
    RespuestaLote,
    RespuestaPrediccion,
    Salud,
    Sismo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api-tsunami")

# Rutas relativas al proyecto (nunca rutas absolutas como C:\Users\...)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.pkl"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "model" / "metadata.json"))
FEATURES_POR_DEFECTO = ["magnitude", "depth", "lat", "lon", "mag_type"]

# Aquí vive el modelo en memoria mientras la API está encendida
ARTIFACTS: dict[str, Any] = {}


# --------------------------------------------------------------------------
# Carga ÚNICA del modelo al iniciar la aplicación (evento lifespan)
# --------------------------------------------------------------------------
def cargar_artefactos() -> None:
    try:
        modelo = joblib.load(MODEL_PATH)
        if not hasattr(modelo, "predict_proba"):
            raise TypeError("El objeto cargado no tiene predict_proba().")
        ARTIFACTS["model"] = modelo
        logger.info("Modelo cargado desde %s", MODEL_PATH)
    except Exception:  # noqa: BLE001
        # La API sigue viva (para que /health informe el problema), pero no predice.
        logger.exception("No se pudo cargar el modelo desde %s", MODEL_PATH)
        return

    try:
        ARTIFACTS["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("No se encontró %s; se usarán valores por defecto.", METADATA_PATH)
        ARTIFACTS["metadata"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_artefactos()
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="API de Alerta de Tsunami — Chile",
    description=(
        "Servicio de inferencia que clasifica si un sismo corresponde a un evento con "
        "**alerta de tsunami** (1) o **sin alerta** (0), usando un modelo RandomForest "
        "entrenado con el catálogo público de USGS.\n\n"
        "⚠️ Proyecto académico: **no** reemplaza la información oficial del SHOA ni de SENAPRED."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Manejo de errores con mensajes claros (sin exponer trazas internas)
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def error_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    errores = [
        {
            "campo": ".".join(str(p) for p in err["loc"] if p != "body") or "body",
            "mensaje": err["msg"],
            "tipo": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Los datos enviados no son válidos. Revisa 'errores'.", "errores": errores},
    )


@app.exception_handler(Exception)
async def error_inesperado(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor."})


# --------------------------------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------------------------------
def obtener_modelo():
    if "model" not in ARTIFACTS:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Revisa que exista model/model.pkl (ejecuta python train.py).",
        )
    return ARTIFACTS["model"]


def version_modelo() -> str:
    return ARTIFACTS.get("metadata", {}).get("model_version", app.version)


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def predecir(sismos: list[Sismo]) -> list[Prediccion]:
    """Convierte los sismos a DataFrame, ejecuta el pipeline y arma las respuestas."""
    modelo = obtener_modelo()
    columnas = ARTIFACTS.get("metadata", {}).get("features", FEATURES_POR_DEFECTO)
    df = pd.DataFrame([s.model_dump() for s in sismos], columns=columnas)

    try:
        clases = [int(c) for c in modelo.classes_]
        preds = modelo.predict(df)
        probas = modelo.predict_proba(df)
    except Exception:  # noqa: BLE001
        logger.exception("Fallo del modelo al predecir")
        raise HTTPException(status_code=500, detail="Error interno al generar la predicción.")

    idx_alerta = clases.index(1)
    resultados = []
    for pred, fila in zip(preds, probas):
        pred = int(pred)
        confianza = float(fila[clases.index(pred)])
        resultados.append(
            Prediccion(
                prediccion=pred,
                etiqueta="ALERTA DE TSUNAMI" if pred == 1 else "SIN ALERTA",
                probabilidad_alerta=round(float(fila[idx_alerta]), 4),
                confianza=round(confianza, 4),
                confianza_pct=f"{confianza * 100:.2f}%",
            )
        )
    return resultados


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", tags=["General"], summary="Bienvenida")
def raiz() -> dict:
    return {
        "mensaje": "API de Alerta de Tsunami — Chile. Visita /docs para probarla.",
        "model_loaded": "model" in ARTIFACTS,
        "endpoints": ["/health", "/model-info", "/predict", "/predict-batch", "/docs"],
    }


@app.get(
    "/health",
    tags=["General"],
    summary="Estado del servicio",
    response_model=Salud,
    responses={503: {"description": "El servicio está arriba pero el modelo no se cargó."}},
)
def health():
    cargado = "model" in ARTIFACTS
    cuerpo = Salud(status="ok" if cargado else "degradado", model_loaded=cargado,
                   model_version=version_modelo() if cargado else None)
    if not cargado:
        return JSONResponse(status_code=503, content=cuerpo.model_dump())
    return cuerpo


@app.get("/model-info", tags=["Modelo"], summary="Metadatos del modelo", response_model=InfoModelo)
def model_info() -> InfoModelo:
    modelo = obtener_modelo()
    meta = ARTIFACTS.get("metadata", {})
    return InfoModelo(
        model_version=version_modelo(),
        estimator=type(modelo.steps[-1][1]).__name__ if hasattr(modelo, "steps") else type(modelo).__name__,
        pipeline_steps=[nombre for nombre, _ in getattr(modelo, "steps", [])],
        features=meta.get("features", FEATURES_POR_DEFECTO),
        categorical_features=meta.get("categorical_features", ["mag_type"]),
        sklearn_version_entrenamiento=meta.get("sklearn"),
        sklearn_version_servidor=sklearn.__version__,
        trained_at=meta.get("trained_at"),
        metrics_test=meta.get("metrics_test", {}),
        metrics_cv=meta.get("metrics_cv", {}),
        dataset=meta.get("dataset", {}),
    )


@app.post("/predict", tags=["Predicción"], summary="Predice UN sismo", response_model=RespuestaPrediccion)
def predict(sismo: Sismo) -> RespuestaPrediccion:
    resultado = predecir([sismo])[0]
    return RespuestaPrediccion(**resultado.model_dump(), model_version=version_modelo(), timestamp=ahora())


@app.post("/predict-batch", tags=["Predicción"], summary="Predice VARIOS sismos", response_model=RespuestaLote)
def predict_batch(lote: LoteSismos) -> RespuestaLote:
    resultados = predecir(lote.sismos)  # mismo orden que la entrada
    return RespuestaLote(n=len(resultados), resultados=resultados, model_version=version_modelo(), timestamp=ahora())
