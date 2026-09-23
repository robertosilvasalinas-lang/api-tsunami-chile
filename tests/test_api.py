"""
Pruebas automatizadas de la API.  Ejecutar desde la raíz del proyecto:
    pytest -v
(Requiere haber ejecutado antes: python train.py)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import ARTIFACTS, MODEL_PATH, app

SISMO_VALIDO = {"magnitude": 7.8, "depth": 20.0, "lat": -22.5, "lon": -70.9, "mag_type": "mww"}
SISMO_LEVE = {"magnitude": 4.6, "depth": 110.0, "lat": -30.1, "lon": -69.5, "mag_type": "mb"}


@pytest.fixture(scope="module")
def client():
    assert MODEL_PATH.exists(), "Falta model/model.pkl: ejecuta primero 'python train.py'"
    with TestClient(app) as c:  # el 'with' activa el lifespan → carga el modelo
        yield c


# ---------------------------- Servicio ----------------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["features"] == ["magnitude", "depth", "lat", "lon", "mag_type"]
    assert body["estimator"] == "RandomForestClassifier"


# ---------------------------- Predicción ----------------------------
def test_predict_exitoso(client):
    r = client.post("/predict", json=SISMO_VALIDO)
    assert r.status_code == 200
    body = r.json()
    assert body["prediccion"] in (0, 1)
    assert 0.0 <= body["probabilidad_alerta"] <= 1.0
    assert 0.5 <= body["confianza"] <= 1.0
    assert "model_version" in body and "timestamp" in body


def test_predict_sin_mag_type_usa_valor_por_defecto(client):
    sismo = {k: v for k, v in SISMO_VALIDO.items() if k != "mag_type"}
    assert client.post("/predict", json=sismo).status_code == 200


def test_predict_batch_mantiene_orden(client):
    lote = [SISMO_VALIDO, SISMO_LEVE, SISMO_VALIDO]
    r = client.post("/predict-batch", json={"sismos": lote})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3
    # Cada resultado del lote debe coincidir con la predicción individual
    for sismo, res in zip(lote, body["resultados"]):
        individual = client.post("/predict", json=sismo).json()
        assert res["prediccion"] == individual["prediccion"]
        assert res["probabilidad_alerta"] == individual["probabilidad_alerta"]


# ---------------------------- Errores (422) ----------------------------
def test_campo_faltante_da_422(client):
    r = client.post("/predict", json={"magnitude": 7.0, "depth": 20.0, "lat": -30.0})
    assert r.status_code == 422
    assert any(e["campo"] == "lon" for e in r.json()["errores"])


def test_tipo_incorrecto_da_422(client):
    r = client.post("/predict", json={**SISMO_VALIDO, "magnitude": "siete"})
    assert r.status_code == 422


def test_fuera_de_rango_da_422(client):
    r = client.post("/predict", json={**SISMO_VALIDO, "magnitude": 12.5})
    assert r.status_code == 422


def test_lote_vacio_da_422(client):
    assert client.post("/predict-batch", json={"sismos": []}).status_code == 422


# ---------------------------- Robustez ----------------------------
def test_sin_modelo_responde_503(client):
    modelo = ARTIFACTS.pop("model")
    try:
        assert client.post("/predict", json=SISMO_VALIDO).status_code == 503
        assert client.get("/health").status_code == 503
    finally:
        ARTIFACTS["model"] = modelo
