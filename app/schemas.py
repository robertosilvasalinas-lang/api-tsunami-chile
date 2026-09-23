"""
schemas.py — "Contratos" de la API definidos con Pydantic.

Pydantic revisa automáticamente cada JSON que llega: si falta un campo, si el tipo
es incorrecto o si un valor está fuera de rango, FastAPI responde 422 sin que el
modelo llegue a ejecutarse.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

EJEMPLO_SISMO = {"magnitude": 7.8, "depth": 20.0, "lat": -22.5, "lon": -70.9, "mag_type": "mww"}


# ------------------------------- ENTRADA -------------------------------
class Sismo(BaseModel):
    """Datos de UN sismo a evaluar."""

    model_config = ConfigDict(
        strict=True,  # "7.5" (texto) se rechaza; 7.5 o 7 (número) se aceptan
        extra="forbid",  # campos desconocidos se rechazan
        json_schema_extra={"examples": [EJEMPLO_SISMO]},
    )

    magnitude: float = Field(..., ge=0, le=10, allow_inf_nan=False, description="Magnitud del sismo (0 a 10).")
    depth: float = Field(..., ge=0, le=700, allow_inf_nan=False, description="Profundidad del hipocentro en km (0 a 700).")
    lat: float = Field(..., ge=-60, le=-15, allow_inf_nan=False, description="Latitud (región de Chile: -60 a -15).")
    lon: float = Field(..., ge=-82, le=-64, allow_inf_nan=False, description="Longitud (región de Chile: -82 a -64).")
    mag_type: str = Field(
        default="mww",
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z]+$",
        description="Tipo de magnitud reportado por la red sísmica (ej.: mww, mb, mwr). Opcional.",
    )

    @field_validator("mag_type")
    @classmethod
    def normalizar_mag_type(cls, v: str) -> str:
        return v.strip().lower()


class LoteSismos(BaseModel):
    """Lista de sismos para predicción por lote."""

    model_config = ConfigDict(extra="forbid")

    sismos: list[Sismo] = Field(..., min_length=1, max_length=1000, description="Entre 1 y 1000 sismos.")


# ------------------------------- SALIDA --------------------------------
class Prediccion(BaseModel):
    prediccion: int = Field(..., description="0 = sin alerta, 1 = alerta de tsunami.")
    etiqueta: str = Field(..., description="Texto legible de la predicción.")
    probabilidad_alerta: float = Field(..., description="Probabilidad (0 a 1) de la clase ALERTA.")
    confianza: float = Field(..., description="Probabilidad (0 a 1) de la clase predicha.")
    confianza_pct: str = Field(..., description="Confianza expresada en porcentaje.")


class RespuestaPrediccion(Prediccion):
    model_config = ConfigDict(protected_namespaces=())  # permite campos que empiezan con "model_"

    model_version: str
    timestamp: str


class RespuestaLote(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # permite campos que empiezan con "model_"

    n: int
    resultados: list[Prediccion]
    model_version: str
    timestamp: str


class Salud(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # permite campos que empiezan con "model_"

    status: str
    model_loaded: bool
    model_version: str | None = None


class InfoModelo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # permite campos que empiezan con "model_"

    model_version: str
    estimator: str
    pipeline_steps: list[str]
    features: list[str]
    categorical_features: list[str]
    sklearn_version_entrenamiento: str | None
    sklearn_version_servidor: str
    trained_at: str | None
    metrics_test: dict
    metrics_cv: dict
    dataset: dict
