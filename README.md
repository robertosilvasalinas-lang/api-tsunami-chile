# 🌊 API de Alerta de Tsunami — Chile

Servicio web (FastAPI) que recibe los datos de un sismo y predice si corresponde a un evento con
**alerta de tsunami (1)** o **sin alerta (0)**, junto con su probabilidad.

> 🔗 **URL pública:** _(completar solo si se hace el despliegue opcional)_
> ⚠️ Proyecto académico. No reemplaza la información oficial del SHOA ni de SENAPRED.

**Curso:** Cloud Computing — Diploma en Data Science, UAI · **Equipo:** _Héctor Cifuentes, Pablo Parra, Matías Sagarra y Roberto Silva_

---

## 1. Problema y datos

- **Tipo de problema:** clasificación binaria supervisada.
- **Dataset:** catálogo sísmico público de USGS para Chile, **2010–2025, magnitud ≥ 5.0**:
  **1.584 sismos**, de los cuales **22 (1,39 %)** tienen alerta de tsunami
  (detalle y cita en [`data/README.md`](data/README.md)).
- **Variables de entrada (5):** `magnitude`, `depth`, `lat`, `lon` (numéricas) y `mag_type` (categórica,
  8 valores observados: mww, mb, mwc, mwr, mwb, mw, ms, ml).
- **Variable objetivo:** `tsunami`, el flag que publica USGS para eventos con información de alerta de tsunami.

**Por qué la ventana empieza en 2010.** Primero probamos con datos desde 1990. La exploración por
quinquenio mostró que **todos los eventos anteriores a 2010 tenían el flag en 0**, incluso sismos de
gran magnitud: USGS no registra ese indicador para eventos antiguos. Esas 1.125 filas habrían
enseñado al modelo un patrón falso ("sismos grandes sin alerta"), así que se descartaron. `train.py`
imprime esa tabla por quinquenio como evidencia de la decisión.

**Por qué magnitud ≥ 5.0.** Los sismos con alerta son siempre eventos grandes. Incluir los de
magnitud 4.5 a 5.0 solo agregaba miles de casos negativos irrelevantes y empeoraba el desbalance.

**Por qué no usamos la API de Chile Alerta.** Su endpoint de prueba devolvió `404` y solo entrega los
últimos sismos, sin etiqueta real. Definir nosotros mismos la etiqueta (por ejemplo "magnitud ≥ 6.5 y
profundidad ≤ 45 km") habría hecho que el modelo se limitara a memorizar nuestra propia regla.

## 2. Modelo

Un único `Pipeline` de scikit-learn, serializado completo en `model/model.pkl`:

```
ColumnTransformer
 ├─ numéricas  → SimpleImputer(mediana) → StandardScaler
 └─ categórica → SimpleImputer("desconocido") → OneHotEncoder(categorías raras agrupadas)
RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
```

**Validación:** división train/test 80/20 estratificada con semilla 42 (1.267 / 317 filas) y, sobre
el conjunto de entrenamiento, **validación cruzada estratificada de 5 particiones repetida 5 veces**.
El conjunto de prueba no se usa hasta la evaluación final, por lo que no hay fuga de información.

Con solo 22 casos positivos en total, el conjunto de prueba contiene apenas 4. Por eso **la
validación cruzada repetida (25 evaluaciones) es nuestra referencia principal** y las métricas de
prueba se reportan como comprobación secundaria.

**Métricas elegidas y por qué.** Las clases están muy desbalanceadas (1,39 % de positivos), así que
la *accuracy* engaña: un modelo que siempre dijera "sin alerta" acertaría el 98,6 % de las veces y
sería inútil. Reportamos:
- **Recall de alerta:** de los sismos que sí tenían alerta, cuántos detectamos.
- **Precisión de alerta:** de las alertas emitidas, cuántas eran correctas.
- **F1 de alerta y F1 macro:** equilibrio entre ambas.
- **PR-AUC:** adecuada para clases desbalanceadas, evalúa las probabilidades sin depender del umbral.

### Resultados

**Referencia principal — validación cruzada (5 particiones × 5 repeticiones, entrenamiento):**

| Métrica | Valor |
|---|---|
| Recall de alerta (media) | **0,947** |
| F1 de alerta (media ± desv.) | **0,621 ± 0,149** |
| PR-AUC (media) | **0,515** |

**Comprobación secundaria — conjunto de prueba (317 sismos, solo 4 con alerta):**

| Métrica (prueba) | Modelo | Regla simple |
|---|---|---|
| Recall de alerta | **1,000** | 0,500 |
| Precisión de alerta | **0,364** | 0,286 |
| F1 de alerta | **0,533** | 0,364 |
| F1 macro | **0,761** | 0,676 |

Matriz de confusión del modelo en prueba: 306 verdaderos negativos, 7 falsos positivos,
0 falsos negativos, 4 verdaderos positivos.

**Línea base:** la regla "magnitud ≥ 6,5 y profundidad ≤ 45 km", que es la heurística habitual y la
que usamos en un prototipo anterior.

### Interpretación

**El modelo supera a la regla simple en las cuatro métricas de prueba.** La diferencia importante
está en el recall: la regla dejó pasar 2 de los 4 sismos con alerta, mientras que el modelo detectó
los 4. En un sistema de alerta temprana ese es el error crítico, porque no avisar de un tsunami real
es mucho más costoso que revisar una falsa alarma. La validación cruzada confirma la tendencia con un
recall medio de 0,947.

**El costo de esa sensibilidad son las falsas alarmas.** Con una precisión de 0,364, el modelo emitió
11 alertas y solo 4 eran correctas. Es un intercambio deliberado, reforzado por el parámetro
`class_weight="balanced"`, que hace que equivocarse en un caso de alerta penalice mucho más que
equivocarse en uno sin alerta.

**Qué no debe leerse de estos resultados.** En el conjunto de prueba, ROC-AUC y PR-AUC dieron 1,000.
Eso no significa un modelo perfecto: con solo 4 casos positivos, basta que reciban mayor probabilidad
que los 313 negativos para obtener ese valor. El número realista es el PR-AUC de la validación
cruzada, 0,515. La desviación de ±0,149 en el F1 confirma que las cifras varían bastante según cómo
se dividan los datos.

**Qué aprendió el modelo, en el fondo.** El flag de USGS responde en gran medida a magnitud, ubicación
oceánica y profundidad, que son justamente nuestras variables. Que el modelo supere a la regla fija
sugiere que aprovecha la interacción entre ellas (por ejemplo, que el umbral de magnitud relevante
cambia según la distancia a la costa) en lugar de aplicar un corte único.

## 3. Estructura

```
├── app/
│   ├── main.py          # aplicación FastAPI (endpoints, carga del modelo, errores)
│   └── schemas.py       # modelos Pydantic de entrada y salida
├── model/
│   ├── model.pkl        # pipeline serializado (lo genera train.py)
│   └── metadata.json    # versiones, features y métricas (lo genera train.py)
├── data/README.md       # origen del dataset
├── docs/                # evidencia de ejecución local
├── tests/test_api.py    # pruebas automatizadas
├── train.py             # entrenamiento y serialización
├── predict.py           # verificación de carga del .pkl en un proceso independiente
├── generar_evidencia.py # guarda la transcripción de llamadas en docs/
├── requirements.txt · runtime.txt · Procfile · Dockerfile · pytest.ini · .gitignore
```

## 4. Cómo reproducirlo (Windows PowerShell)

Requisito: **Python 3.14.2** (misma versión de `runtime.txt` y del `Dockerfile`).

```powershell
git clone <URL-del-repositorio>
cd api-tsunami-chile
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py          # descarga datos, entrena y crea model/model.pkl (opcional: ya viene en el repo)
python predict.py        # verifica que el .pkl carga en otro proceso
uvicorn app.main:app --reload --port 8000
```

En macOS/Linux, activar con `source .venv/bin/activate`.
Abrir **http://localhost:8000/docs** para probar la API desde el navegador.

## 5. Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/` | Bienvenida y lista de endpoints |
| GET | `/health` | 200 si el modelo está cargado; 503 si no |
| GET | `/model-info` | Tipo de estimador, variables, métricas, versiones |
| POST | `/predict` | Predice un sismo |
| POST | `/predict-batch` | Predice una lista de sismos (mismo orden) |
| GET | `/docs` | Documentación interactiva Swagger |

**Ejemplo** (en PowerShell usar `curl.exe`, no `curl`):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{\"magnitude\": 7.8, \"depth\": 20, \"lat\": -22.5, \"lon\": -70.9, \"mag_type\": \"mww\"}'
```

Respuesta (los números dependen del entrenamiento):

```json
{"prediccion": 1, "etiqueta": "ALERTA DE TSUNAMI", "probabilidad_alerta": 0.81,
 "confianza": 0.81, "confianza_pct": "80.58%", "model_version": "1.0.0",
 "timestamp": "2026-09-22T00:29:19+00:00"}
```

**Reglas de validación:** `magnitude` 0–10, `depth` 0–700 km, `lat` −60 a −15, `lon` −82 a −64,
`mag_type` opcional (por defecto `mww`). Campos faltantes, tipos incorrectos (ej. `"7.5"` como texto),
valores fuera de rango o campos desconocidos → **422** con la lista de errores.
Fallo del modelo → **500** con mensaje genérico (sin trazas internas). Modelo no cargado → **503**.

**Evidencia:** captura de `/docs` en `docs/evidencia_docs.png` y transcripción de llamadas en
[`docs/evidencia_llamadas.md`](docs/evidencia_llamadas.md).

## 6. Pruebas automatizadas

```powershell
pytest -v
```

Salida:

```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.1.1, pluggy-1.6.0 -- B:\MAGISTER DATA SCIENCE\CLOUD COMPUTING\PROYECTO\api-tsunami-chile\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: B:\MAGISTER DATA SCIENCE\CLOUD COMPUTING\PROYECTO\api-tsunami-chile
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 10 items

tests/test_api.py::test_health_ok PASSED                                 [ 10%]
tests/test_api.py::test_model_info PASSED                                [ 20%]
tests/test_api.py::test_predict_exitoso PASSED                           [ 30%]
tests/test_api.py::test_predict_sin_mag_type_usa_valor_por_defecto PASSED [ 40%]
tests/test_api.py::test_predict_batch_mantiene_orden PASSED              [ 50%]
tests/test_api.py::test_campo_faltante_da_422 PASSED                     [ 60%]
tests/test_api.py::test_tipo_incorrecto_da_422 PASSED                    [ 70%]
tests/test_api.py::test_fuera_de_rango_da_422 PASSED                     [ 80%]
tests/test_api.py::test_lote_vacio_da_422 PASSED                         [ 90%]
tests/test_api.py::test_sin_modelo_responde_503 PASSED                   [100%]

============================= 10 passed in 8.70s ==============================


> En nuestra ejecución: **10 passed**.

## 7. Decisiones técnicas

- **Pipeline completo en el `.pkl`:** la API no reimplementa ninguna transformación (evita *training–serving skew*).
- **Carga única del modelo** con el evento `lifespan` de FastAPI, no en cada petición.
- **Si el modelo no carga, la API no se cae:** `/health` responde 503 y lo informa, útil para diagnosticar en la nube.
- **`class_weight="balanced"`** para compensar el desbalance de clases.
- **`mag_type` opcional** para que un cliente que solo conoce magnitud, profundidad y coordenadas pueda usar la API.
- **Versiones fijadas** en `requirements.txt`; la versión de scikit-learn usada queda registrada en `metadata.json` y visible en `/model-info`.

## 8. Limitaciones

- El flag `tsunami` de USGS indica condiciones para emitir información de tsunami, no que un tsunami ocurrió.
- Solo 22 casos con alerta en todo el dataset: las métricas tienen varianza alta (±0,149 en el F1 de
  validación cruzada) y el conjunto de prueba contiene apenas 4 positivos.
- La ventana de datos se limita a 2010–2025 porque USGS no registra el flag antes de esa fecha, lo que
  deja fuera eventos históricos relevantes como el terremoto de Maule (2010, magnitud 8,8).
- El modelo no considera la distancia real a la fosa ni el mecanismo focal del sismo.

## 9. Despliegue en la nube (opcional)

_Completar si se realiza: proveedor, pasos, variables de entorno, problema encontrado y su solución,
y un `curl` contra la URL pública con su respuesta._
