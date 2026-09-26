# Evidencia de llamadas a la API

Servidor: `https://api-tsunami-chile.onrender.com` — generado el 2026-09-26 09:47

## Estado del servicio

`GET /health` → **HTTP 200**

Respuesta:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

## Metadatos del modelo

`GET /model-info` → **HTTP 200**

Respuesta:
```json
{
  "model_version": "1.0.0",
  "estimator": "RandomForestClassifier",
  "pipeline_steps": [
    "pre",
    "clf"
  ],
  "features": [
    "magnitude",
    "depth",
    "lat",
    "lon",
    "mag_type"
  ],
  "categorical_features": [
    "mag_type"
  ],
  "sklearn_version_entrenamiento": "1.9.1",
  "sklearn_version_servidor": "1.9.1",
  "trained_at": "2026-09-22T23:03:23+00:00",
  "metrics_test": {
    "precision_alerta": 0.3636,
    "recall_alerta": 1.0,
    "f1_alerta": 0.5333,
    "f1_macro": 0.761,
    "roc_auc": 1.0,
    "pr_auc": 1.0
  },
  "metrics_cv": {
    "cv_f1_alerta_media": 0.6214,
    "cv_f1_alerta_std": 0.1486,
    "cv_pr_auc_media": 0.515,
    "cv_recall_alerta_media": 0.9467
  },
  "dataset": {
    "fuente": "USGS Earthquake Catalog (ComCat), API FDSN Event",
    "url": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "parametros": {
      "format": "geojson",
      "starttime": "2010-01-01",
      "endtime": "2026-01-01",
      "minlatitude": -56,
      "maxlatitude": -17,
      "minlongitude": -80,
      "maxlongitude": -66,
      "minmagnitude": 5.0,
      "orderby": "time-asc",
      "limit": 20000
    },
    "n_filas": 1584,
    "n_positivos": 22,
    "tasa_positivos": 0.0139
  }
}
```

## Predicción exitosa

`POST /predict` → **HTTP 200**

Petición:
```json
{
  "magnitude": 7.8,
  "depth": 20.0,
  "lat": -22.5,
  "lon": -70.9,
  "mag_type": "mww"
}
```

Respuesta:
```json
{
  "prediccion": 1,
  "etiqueta": "ALERTA DE TSUNAMI",
  "probabilidad_alerta": 0.9539,
  "confianza": 0.9539,
  "confianza_pct": "95.39%",
  "model_version": "1.0.0",
  "timestamp": "2026-09-26T12:47:43+00:00"
}
```

## Predicción por lote

`POST /predict-batch` → **HTTP 200**

Petición:
```json
{
  "sismos": [
    {
      "magnitude": 7.8,
      "depth": 20.0,
      "lat": -22.5,
      "lon": -70.9,
      "mag_type": "mww"
    },
    {
      "magnitude": 4.6,
      "depth": 110.0,
      "lat": -30.1,
      "lon": -69.5,
      "mag_type": "mb"
    }
  ]
}
```

Respuesta:
```json
{
  "n": 2,
  "resultados": [
    {
      "prediccion": 1,
      "etiqueta": "ALERTA DE TSUNAMI",
      "probabilidad_alerta": 0.9539,
      "confianza": 0.9539,
      "confianza_pct": "95.39%"
    },
    {
      "prediccion": 0,
      "etiqueta": "SIN ALERTA",
      "probabilidad_alerta": 0.0,
      "confianza": 1.0,
      "confianza_pct": "100.00%"
    }
  ],
  "model_version": "1.0.0",
  "timestamp": "2026-09-26T12:47:45+00:00"
}
```

## Entrada inválida (falta 'lon' y magnitud fuera de rango) → 422

`POST /predict` → **HTTP 422**

Petición:
```json
{
  "magnitude": 12.5,
  "depth": 20.0,
  "lat": -30.0
}
```

Respuesta:
```json
{
  "detail": "Los datos enviados no son válidos. Revisa 'errores'.",
  "errores": [
    {
      "campo": "magnitude",
      "mensaje": "Input should be less than or equal to 10",
      "tipo": "less_than_equal"
    },
    {
      "campo": "lon",
      "mensaje": "Field required",
      "tipo": "missing"
    }
  ]
}
```
