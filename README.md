# 🌊 API de Alerta de Tsunami — Chile

Servicio web (FastAPI) que recibe los datos de un sismo y predice si corresponde a un evento con
**alerta de tsunami (1)** o **sin alerta (0)**, junto con su probabilidad.

**URL pública:** https://api-tsunami-chile.onrender.com
> ⚠️ Proyecto académico. No reemplaza la información oficial del SHOA ni de SENAPRED.

**Curso:** Cloud Computing — Diploma en Data Science, UAI · **Equipo:** _Héctor Cifuentes, Pablo Parra y Roberto Silva_

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
├── .github/workflows/tests.yml # CI: corre pytest en cada push/PR a main
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
| GET | `/` | Página de bienvenida (HTML) con menú de acceso a `/docs`, `/health` y `/model-info` |
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

### Cómo interpretar una predicción

Las métricas de la sección 2 dicen qué tan bueno es el modelo en conjunto. Esta
sección responde otra pregunta: **qué hacer con una respuesta concreta**.

Una llamada a `/predict` devuelve seis campos:

```json
{
  "prediccion": 1,
  "etiqueta": "ALERTA DE TSUNAMI",
  "probabilidad_alerta": 0.9539,
  "confianza": 0.9539,
  "confianza_pct": "95.39%",
  "model_version": "1.0.0",
  "timestamp": "2026-09-22T21:25:00+00:00"
}
```

| Campo | Qué significa |
|---|---|
| `prediccion` | La clase elegida: `1` con alerta, `0` sin alerta. |
| `etiqueta` | La misma clase en palabras, para no tener que recordar el código. |
| `probabilidad_alerta` | **Siempre** la probabilidad de la clase 1. Es el número a mirar si quieres comparar sismos entre sí. |
| `confianza` | La probabilidad de la clase **que el modelo eligió**. |
| `model_version` | Qué versión del modelo respondió. Sirve para rastrear una predicción si mañana se reentrena. |
| `timestamp` | Momento UTC de la respuesta. |

**`probabilidad_alerta` y `confianza` no son lo mismo**, aunque coincidan en el
ejemplo de arriba. Coinciden solo cuando la predicción es 1. Cuando el modelo
responde "sin alerta", `confianza` pasa a ser la probabilidad de la clase 0:

| Sismo | `probabilidad_alerta` | `prediccion` | `confianza` |
|---|---|---|---|
| Magnitud 7,8 · 20 km · costa de Antofagasta | 0,954 | 1 | 0,954 |
| Magnitud 4,6 · 110 km · cordillera | 0,020 | 0 | 0,980 |

En el segundo caso, un 98 % de confianza **no** es un 98 % de probabilidad de
tsunami: es lo contrario. Leer `confianza` sin mirar `prediccion` es el error
más fácil de cometer con esta API.

**El umbral es 0,5 y es una decisión, no una ley.** El modelo responde `1`
cuando `probabilidad_alerta ≥ 0,5`. Como la API entrega siempre la probabilidad
cruda, quien consuma el servicio puede aplicar su propio corte sin reentrenar
nada: exigir 0,8 para reducir falsas alarmas, o bajar a 0,3 si prefiere
sobre-avisar.

#### Las dos respuestas no pesan lo mismo

Este es el punto que conviene tener claro antes de usar el servicio para algo.
Con recall de 0,947 en validación cruzada y precisión de 0,364 en prueba:

- **Un `SIN ALERTA` es una respuesta fuerte.** El modelo casi no deja pasar
  eventos con alerta, así que cuando dice que no, lo más probable es que no.
- **Una `ALERTA` es una respuesta débil.** De las 11 alertas que emitió en el
  conjunto de prueba, solo 4 eran correctas. Aproximadamente dos de cada tres
  son falsas alarmas.

Esa asimetría es deliberada: `class_weight="balanced"` hace que no detectar un
evento real salga mucho más caro que revisar una alarma de más. Para un sistema
de alerta temprana es el intercambio correcto, pero significa que **una alerta
del modelo es una señal para verificar, no una conclusión**.

#### Lo que la predicción no dice

- No dice que vaya a ocurrir un tsunami. El objetivo aprendido es el flag
  `tsunami` de USGS, que marca eventos con información del sistema de alerta de
  NOAA — condiciones para emitir información, no un tsunami confirmado.
- No estima altura de ola, hora de llegada ni zona afectada.
- No reemplaza al SHOA ni a SENAPRED. Este es un ejercicio académico.

#### Cuándo desconfiar de una respuesta

El modelo vio 1.584 sismos chilenos de magnitud ≥ 5,0 entre 2010 y 2025. Fuera
de ese marco extrapola sin avisar:

- Coordenadas fuera de Chile. La validación acepta latitud −60 a −15 y longitud
  −82 a −64, pero dentro de ese rectángulo hay zonas con muy pocos datos.
- Magnitudes bajo 5,0, que quedaron excluidas del entrenamiento a propósito.
- Un `mag_type` que no esté entre los siete observados (`mb`, `ml`, `mw`,
  `mwb`, `mwc`, `mwr`, `mww`). El `OneHotEncoder` lo tolera sin caerse, pero la
  predicción se apoya solo en las variables numéricas.

Para ver qué variables y qué categorías conoce el modelo que está respondiendo,
consulta `GET /model-info`.

## 6. Pruebas automatizadas

```powershell
pytest -v
```

Salida:

```
============================= test session starts =============================
platform darwin -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: api-tsunami-chile
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 14 items

tests/test_api.py::test_raiz_muestra_menu_html PASSED                    [  7%]
tests/test_api.py::test_health_ok PASSED                                 [ 14%]
tests/test_api.py::test_health_html_para_navegador PASSED                [ 21%]
tests/test_api.py::test_model_info PASSED                                [ 28%]
tests/test_api.py::test_model_info_html_para_navegador PASSED            [ 35%]
tests/test_api.py::test_predict_exitoso PASSED                           [ 42%]
tests/test_api.py::test_predict_sin_mag_type_usa_valor_por_defecto PASSED [ 50%]
tests/test_api.py::test_predict_batch_mantiene_orden PASSED              [ 57%]
tests/test_api.py::test_campo_faltante_da_422 PASSED                     [ 64%]
tests/test_api.py::test_tipo_incorrecto_da_422 PASSED                    [ 71%]
tests/test_api.py::test_fuera_de_rango_da_422 PASSED                     [ 78%]
tests/test_api.py::test_lote_vacio_da_422 PASSED                         [ 85%]
tests/test_api.py::test_sin_modelo_responde_503 PASSED                   [ 92%]
tests/test_api.py::test_fallo_del_modelo_responde_500 PASSED             [100%]

============================== 14 passed in 6.05s ==============================


> En nuestra ejecución: **14 passed** (cobertura de línea de `app/`: 94%).

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

## 9. Despliegue en la nube

**URL pública:** https://api-tsunami-chile.onrender.com

| | |
|---|---|
| Estado del servicio | https://api-tsunami-chile.onrender.com/health |
| Documentación interactiva | https://api-tsunami-chile.onrender.com/docs |
| Proveedor | Render — Web Service, plan Free |
| Runtime | Docker |

### Configuración

| Campo | Valor |
|---|---|
| Language | Docker |
| Dockerfile Path | `./Dockerfile` |
| Branch | `main` |
| Instance Type | Free |
| Health Check Path | `/health` |
| Auto-Deploy | AJUSTAR: On / no disponible por esta vía |

### Variables de entorno

Ninguna propia del proyecto. Render inyecta `PORT` automáticamente y el `CMD`
del Dockerfile lo lee con `--port ${PORT}`; en el despliegue real asignó el
puerto 10000. El servicio no usa credenciales ni consulta APIs externas en
tiempo de ejecución: el modelo entrenado viaja dentro de la imagen.

### Problemas encontrados y sus soluciones

**1. Elección del runtime: Docker en vez de Python.**
El proyecto requiere Python 3.14.2, la versión con la que se entrenó el modelo
y la que declara `runtime.txt`. El runtime nativo de Python de Render no
permite elegir la versión desde la interfaz —solo ofrece «Python 3»— y la
deduce al construir, quedando sujeta a lo que la plataforma tenga disponible.
Desplegar por esa vía arriesgaba cargar el `.pkl` con un intérprete y un
scikit-learn distintos de los que lo generaron, que es exactamente el fallo
que el enunciado penaliza.

*Solución:* se desplegó con el runtime **Docker**, usando el `Dockerfile` del
repositorio, que parte de la imagen `python:3.14.2-slim`. La versión del
intérprete viaja dentro de la imagen y deja de depender de la plataforma.
Efecto secundario a tener presente: en modo Docker el `Procfile` queda sin
usar, porque el comando de arranque lo define el `CMD` del Dockerfile. Ambos
declaran lo mismo, así que no hay conflicto.

**2. El primer despliegue falló en 7 segundos.**
El log mostró:

```
==> Cloning from https://github.com/pabloparrastuardo-bot/api-siniestros-chile
error: failed to solve: failed to read dockerfile:
       open Dockerfile: no such file or directory
```

*Solución:* la primera línea del log delató la causa: el servicio estaba
clonando otro repositorio, que no tiene Dockerfile. Se había seleccionado el
repo equivocado en la lista. Como Render no permite cambiar el repositorio de
un servicio ya creado, se eliminó el servicio y se creó uno nuevo. Lección
práctica: en un despliegue fallido, la primera línea del log suele decir más
que el mensaje de error final.

**3. Render no podía acceder al repositorio del equipo.**
Render se integra con GitHub mediante una *GitHub App*, y una GitHub App se
instala en una **cuenta**. El repositorio pertenece a la cuenta personal de un
integrante del equipo: ser colaborador permite escribir en él, pero no
instalar aplicaciones en la cuenta ajena. Por eso el repositorio no aparecía
en la lista de Render y la opción de conceder acceso no existía —el selector
de cuentas solo mostraba la cuenta propia.

*Solución:* se usó la pestaña **Public Git Repository** de Render, que clona
un repositorio público directamente por URL, sin pasar por los permisos de la
GitHub App. Es viable porque el repositorio es público.

### Evidencia

Llamada real contra la URL pública, ejecutada desde Swagger en producción:

```bash
curl -X POST 'https://api-tsunami-chile.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{"magnitude": 7.8, "depth": 20, "lat": -22.5, "lon": -70.9, "mag_type": "mww"}'
```

Respuesta (HTTP 200):

```json
{
  "prediccion": 1,
  "etiqueta": "ALERTA DE TSUNAMI",
  "probabilidad_alerta": 0.9539,
  "confianza": 0.9539,
  "confianza_pct": "95.39%",
  "model_version": "1.0.0",
  "timestamp": "2026-09-25T13:28:45+00:00"
}
```

Es la **misma probabilidad** que devuelve la ejecución local documentada en
`docs/evidencia_llamadas.md`. Que coincidan confirma que el artefacto
serializado se comporta igual en ambos entornos, que es justamente el objetivo
de haber guardado el pipeline completo en el `.pkl`.

Captura del servicio público en `docs/evidencia_render.png`.

Log del arranque en Render:

```
==> Deploying...
INFO:     Started server process [7]
INFO:     Waiting for application startup.
2026-09-25 13:25:58,006 INFO Modelo cargado desde /app/model/model.pkl
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:10000
==> Your service is live
```

La línea `Modelo cargado desde /app/model/model.pkl` aparece **una sola vez**,
durante el arranque: es la confirmación en producción de que el `lifespan`
carga el modelo al iniciar el proceso y no en cada petición.

### Nota sobre el arranque en frío

El plan gratuito de Render suspende la instancia tras 15 minutos sin tráfico.
La primera petición después de una pausa puede tardar cerca de un minuto
mientras el contenedor vuelve a levantar; las siguientes responden en
milisegundos. Es el costo esperado de un servicio sin instancias mínimas
reservadas, y el motivo por el que conviene visitar `/health` antes de
cualquier demostración.

