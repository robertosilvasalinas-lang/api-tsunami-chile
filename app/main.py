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
from fastapi.responses import HTMLResponse, JSONResponse

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


def _prefiere_html(request: Request) -> bool:
    """El navegador pide 'text/html' explícitamente; curl, pytest y Swagger no."""
    return "text/html" in request.headers.get("accept", "")


# Tema oscuro propio (no depende de ningún host externo): superficies, texto y
# acentos de estado, con la misma pareja fondo-oscuro/texto-claro para cada rol
# (éxito, error, advertencia) para que el contraste sea consistente en toda la app.
_ESTILO = """
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 48px 20px 72px;
      background: #0b0d10; color: #e8eaed;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
    }
    .wrap { max-width: 720px; margin: 0 auto; }
    .nav { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 32px; }
    .brand { display: flex; align-items: center; gap: 8px; font-weight: 500; font-size: 15px; color: #e8eaed; text-decoration: none; }
    .nav-links { display: flex; gap: 18px; }
    .nav-links a { color: #5dcaa5; text-decoration: none; font-size: 14px; }
    .nav-links a:hover { text-decoration: underline; }
    h1 { font-size: 24px; font-weight: 500; margin: 0 0 8px; }
    h2 { font-size: 16px; font-weight: 500; margin: 32px 0 12px; color: #c7cbd1; }
    p.lead { color: #9aa1ab; margin: 0 0 24px; font-size: 15px; max-width: 60ch; }
    .card { background: #14171b; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 4px 22px; margin-bottom: 22px; }
    table { width: 100%; border-collapse: collapse; }
    tr { border-bottom: 1px solid rgba(255,255,255,0.06); }
    tr:last-child { border-bottom: none; }
    th, td { text-align: left; padding: 13px 0; font-size: 14px; font-weight: 400; vertical-align: top; }
    th { width: 42%; color: #9aa1ab; padding-right: 12px; }
    td { color: #e8eaed; }
    .badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px; border-radius: 999px; font-size: 13px; font-weight: 500; }
    .badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: none; }
    .badge-ok { background: #04342c; color: #9fe1cb; }
    .badge-bad { background: #501313; color: #f7c1c1; }
    .badge-warn { background: #412402; color: #fac775; }
    .btn { display: inline-block; margin-top: 4px; padding: 10px 20px; border-radius: 10px; background: #1d9e75; color: #04342c; font-weight: 500; font-size: 14px; text-decoration: none; }
    .btn:hover { background: #5dcaa5; }
    .footer { color: #6b7280; font-size: 13px; margin-top: 28px; }
    .info { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; margin-left: 6px; border-radius: 50%; background: rgba(255,255,255,0.1); color: #9aa1ab; font-size: 11px; font-style: normal; line-height: 1; cursor: help; vertical-align: middle; }
    .info .tooltip { visibility: hidden; opacity: 0; position: absolute; left: 0; top: 130%; background: #1f242b; color: #e8eaed; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 9px 11px; font-size: 12px; font-weight: 400; line-height: 1.45; width: 240px; z-index: 10; transition: opacity .15s; }
    .info:hover .tooltip, .info:focus .tooltip { visibility: visible; opacity: 1; }
"""

_NAV = """
    <nav class="nav">
      <a class="brand" href="/">🌊 API tsunami Chile</a>
      <div class="nav-links">
        <a href="/docs">docs</a>
        <a href="/health">health</a>
        <a href="/model-info">model-info</a>
      </div>
    </nav>
"""


def _pagina(titulo: str, cuerpo: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{titulo} — API de Alerta de Tsunami</title>
  <style>{_ESTILO}</style>
</head>
<body>
  <div class="wrap">
    {_NAV}
    {cuerpo}
    <p class="footer">Proyecto académico. No reemplaza la información oficial del SHOA ni de SENAPRED.</p>
  </div>
</body>
</html>"""


def _badge(texto: str, tipo: str) -> str:
    clase = {"ok": "badge-ok", "bad": "badge-bad", "warn": "badge-warn"}[tipo]
    return f'<span class="badge {clase}">{texto}</span>'


# Traducción de llaves crudas de metadata.json (snake_case) a etiquetas legibles.
# Las llaves que no están aquí (p. ej. las que ya vienen armadas a mano, como
# "Versión del modelo") se muestran sin tocar: solo se reformatean identificadores
# en snake_case puro (ver _etiqueta).
_ETIQUETAS = {
    "fuente": "Fuente",
    "url": "URL",
    "n_filas": "Filas en el dataset",
    "n_positivos": "Casos con alerta",
    "tasa_positivos": "Tasa de positivos",
    "precision_alerta": "Precisión (alerta)",
    "recall_alerta": "Recall (alerta)",
    "f1_alerta": "F1 (alerta)",
    "f1_macro": "F1 macro",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "cv_f1_alerta_media": "F1 alerta — media (CV)",
    "cv_f1_alerta_std": "F1 alerta — desv. estándar (CV)",
    "cv_pr_auc_media": "PR-AUC — media (CV)",
    "cv_recall_alerta_media": "Recall alerta — media (CV)",
}


def _etiqueta(clave: str) -> str:
    if clave in _ETIQUETAS:
        return _ETIQUETAS[clave]
    if "_" in clave and clave == clave.lower():
        texto = clave.replace("_", " ")
        return texto[0].upper() + texto[1:]
    return clave


# Definiciones cortas para el ícono ⓘ, indexadas por la etiqueta ya traducida
# (no por la llave cruda): así cubren tanto las tablas armadas desde metadata.json
# como las armadas a mano (p. ej. "scikit-learn (servidor)").
_GLOSARIO = {
    "Precisión (alerta)": "De las alertas que emitió el modelo, qué porcentaje eran correctas.",
    "Recall (alerta)": "De los sismos que sí tenían alerta, qué porcentaje detectó el modelo.",
    "F1 (alerta)": "Promedio armónico entre precisión y recall de la clase alerta.",
    "F1 macro": "Promedio de F1 entre ambas clases (alerta y sin alerta), sin ponderar por frecuencia.",
    "ROC-AUC": "Capacidad de separar ambas clases sin depender de un umbral. 1.0 = perfecto, 0.5 = azar.",
    "PR-AUC": "Como ROC-AUC, pero centrado en la clase minoritaria (alerta); más útil con clases desbalanceadas.",
    "F1 alerta — media (CV)": "F1 de alerta promediado sobre 25 particiones de validación cruzada (5 folds × 5 repeticiones).",
    "F1 alerta — desv. estándar (CV)": "Qué tanto varía el F1 entre particiones de validación cruzada. Alto = resultados inestables.",
    "PR-AUC — media (CV)": "PR-AUC promediado sobre la validación cruzada.",
    "Recall alerta — media (CV)": "Recall de alerta promediado sobre la validación cruzada.",
    "Filas en el dataset": "Cantidad total de sismos usados para entrenar y evaluar el modelo.",
    "Casos con alerta": "Cuántos de esos sismos tenían el flag de alerta de tsunami de USGS.",
    "Tasa de positivos": "Porcentaje de sismos con alerta sobre el total del dataset.",
    "scikit-learn (servidor)": (
        "La versión de scikit-learn cargada en este servidor debe coincidir con la que "
        "entrenó el modelo, o la predicción puede fallar o dar resultados distintos."
    ),
}


def _tabla(filas: dict) -> str:
    """Renderiza un dict plano como tabla HTML de dos columnas (clave / valor),
    con un ícono ⓘ junto a las etiquetas que tienen definición en _GLOSARIO."""

    def _valor(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v) or "—"
        if isinstance(v, dict):
            return ", ".join(f"{k}={x}" for k, x in v.items()) or "—"
        return "—" if v in (None, "") else str(v)

    def _fila(k: str, v: Any) -> str:
        etiqueta = _etiqueta(k)
        definicion = _GLOSARIO.get(etiqueta)
        icono = (
            f'<span class="info" tabindex="0" aria-label="{definicion}">ⓘ'
            f'<span class="tooltip">{definicion}</span></span>'
            if definicion else ""
        )
        return f"<tr><th>{etiqueta}{icono}</th><td>{_valor(v)}</td></tr>"

    filas_html = "\n".join(_fila(k, v) for k, v in filas.items())
    return f'<div class="card"><table>{filas_html}</table></div>'


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
@app.get("/", tags=["General"], summary="Bienvenida", response_class=HTMLResponse)
def raiz() -> HTMLResponse:
    cargado = "model" in ARTIFACTS
    estado = _badge("modelo cargado", "ok") if cargado else _badge("modelo no disponible", "bad")
    cuerpo = f"""
  <h1>API de alerta de tsunami — Chile</h1>
  <p class="lead">Servicio de inferencia que clasifica sismos en Chile como <strong>con alerta de tsunami</strong>
  o <strong>sin alerta</strong>, junto con su probabilidad. Entrenado con el catálogo público de USGS.</p>
  <p>{estado}</p>
  <p><a class="btn" href="/docs">Probar la API en /docs</a></p>
"""
    return HTMLResponse(content=_pagina("Inicio", cuerpo))


@app.get(
    "/health",
    tags=["General"],
    summary="Estado del servicio",
    response_model=Salud,
    responses={503: {"description": "El servicio está arriba pero el modelo no se cargó."}},
)
def health(request: Request):
    cargado = "model" in ARTIFACTS
    cuerpo = Salud(status="ok" if cargado else "degradado", model_loaded=cargado,
                   model_version=version_modelo() if cargado else None)

    if _prefiere_html(request):
        badge = _badge("ok", "ok") if cargado else _badge("degradado", "bad")
        html_cuerpo = f"""
  <h1>Estado del servicio</h1>
  {_tabla({
        "Estado": badge,
        "Modelo cargado": "Sí" if cargado else "No",
        "Versión del modelo": cuerpo.model_version,
    })}
"""
        return HTMLResponse(content=_pagina("Estado", html_cuerpo), status_code=200 if cargado else 503)

    if not cargado:
        return JSONResponse(status_code=503, content=cuerpo.model_dump())
    return cuerpo


@app.get("/model-info", tags=["Modelo"], summary="Metadatos del modelo", response_model=InfoModelo)
def model_info(request: Request):
    modelo = obtener_modelo()
    meta = ARTIFACTS.get("metadata", {})
    info = InfoModelo(
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

    if _prefiere_html(request):
        coinciden = info.sklearn_version_entrenamiento == info.sklearn_version_servidor
        version_sklearn = (
            f"{info.sklearn_version_servidor} {_badge('coincide con entrenamiento', 'ok')}"
            if coinciden
            else f"{info.sklearn_version_servidor} {_badge(f'entrenado con {info.sklearn_version_entrenamiento}', 'warn')}"
        )
        general = _tabla({
            "Versión del modelo": info.model_version,
            "Estimador": info.estimator,
            "Pasos del pipeline": info.pipeline_steps,
            "Variables de entrada": info.features,
            "Variables categóricas": info.categorical_features,
            "scikit-learn (servidor)": version_sklearn,
            "Entrenado el": info.trained_at,
        })
        dataset_sin_parametros = {k: v for k, v in info.dataset.items() if k != "parametros"}
        cuerpo = f"""
  <h1>Metadatos del modelo</h1>
  {general}
  <h2>Métricas — conjunto de prueba</h2>
  {_tabla(info.metrics_test)}
  <h2>Métricas — validación cruzada</h2>
  {_tabla(info.metrics_cv)}
  <h2>Dataset de entrenamiento</h2>
  {_tabla(dataset_sin_parametros)}
"""
        return HTMLResponse(content=_pagina("Modelo", cuerpo))

    return info


@app.post("/predict", tags=["Predicción"], summary="Predice UN sismo", response_model=RespuestaPrediccion)
def predict(sismo: Sismo) -> RespuestaPrediccion:
    resultado = predecir([sismo])[0]
    return RespuestaPrediccion(**resultado.model_dump(), model_version=version_modelo(), timestamp=ahora())


@app.post("/predict-batch", tags=["Predicción"], summary="Predice VARIOS sismos", response_model=RespuestaLote)
def predict_batch(lote: LoteSismos) -> RespuestaLote:
    resultados = predecir(lote.sismos)  # mismo orden que la entrada
    return RespuestaLote(n=len(resultados), resultados=resultados, model_version=version_modelo(), timestamp=ahora())
