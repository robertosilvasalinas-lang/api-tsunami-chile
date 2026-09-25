# Evidencia de ejecución local

Esta carpeta contiene la prueba de que el servicio se levantó y respondió
correctamente en `localhost`, que es lo que exige la etapa E7 del enunciado.

Cada archivo demuestra algo distinto. Abajo se explica qué muestra cada uno, por
qué sirve como evidencia y cómo regenerarlo.

---

## `evidencia_docs.png`

**Qué es.** Captura de pantalla de `http://localhost:8000/docs` abierto en el
navegador, con el servicio corriendo.

**Qué demuestra.** Tres cosas a la vez:

1. Que el proceso levantó y está escuchando en el puerto 8000.
2. Que FastAPI generó la documentación interactiva Swagger automáticamente, sin
   que nosotros escribiéramos una línea de HTML. Esa página sale de los modelos
   Pydantic de `app/schemas.py`: los rangos válidos de `magnitude`, `depth`,
   `lat` y `lon` que se ven en pantalla son los mismos que rechazan una petición
   inválida con 422.
3. Que los seis endpoints declarados en el README existen de verdad y son
   navegables.

**Cómo regenerarla.**

```powershell
uvicorn app.main:app --reload --port 8000
```

Abrir `http://localhost:8000/docs` y capturar con **Win + Shift + S**. Que quede
visible la barra de direcciones con `localhost:8000`, para que se note que es
una ejecución local y no la versión desplegada.

---

## `evidencia_llamadas.md`

**Qué es.** Transcripción de una sesión real contra la API: cada petición con su
cuerpo, su código de estado y su respuesta completa. La generó
`generar_evidencia.py`, no se escribió a mano.

**Qué demuestra.** El enunciado pide al menos tres llamadas: una predicción
exitosa, una por lote y una entrada inválida que devuelva 422. La transcripción
cubre las tres y agrega dos de contexto:

| Llamada | Qué prueba |
|---|---|
| `GET /health` | El modelo quedó cargado en memoria. Responde `model_loaded: true` y la versión. |
| `GET /model-info` | Las variables, métricas y versiones que declara el modelo servido coinciden con las del README. |
| `POST /predict` — magnitud 7,8 a 20 km | Predicción exitosa: devuelve `ALERTA DE TSUNAMI` con 95,39 % de confianza. |
| `POST /predict` — magnitud 4,6 a 110 km | El modelo discrimina: mismo endpoint, resultado `SIN ALERTA`. |
| `POST /predict-batch` | Varios sismos en una llamada, con las respuestas en el mismo orden de entrada. |
| `POST /predict` con campos faltantes o fuera de rango | Devuelve **422** con la lista de errores por campo, sin llegar al modelo. |

Los dos casos de `/predict` con resultados opuestos son deliberados: una sola
predicción exitosa no prueba que el modelo discrimine, solo que el endpoint
responde. Dos con resultados distintos sí.

**Cómo regenerarla.** Con la API corriendo en otra ventana:

```powershell
python generar_evidencia.py
```

El script escribe este archivo desde cero. Si la API no está levantada, falla
por conexión rechazada y no genera nada — lo cual es sano, porque evita que
quede una evidencia desactualizada de una ejecución anterior.

---

## Por qué la evidencia va al repositorio

Las capturas y transcripciones no son decoración. Quien corrige no ejecuta el
proyecto en su máquina: lee el repositorio. Estos archivos son lo único que
respalda que el servicio funcionó de verdad, y por eso se versionan en vez de
quedar en el escritorio de alguien.

Los tres archivos que sostienen esa afirmación son complementarios:

- `evidencia_docs.png` prueba que **el servicio se levantó**.
- `evidencia_llamadas.md` prueba que **respondió correctamente**, incluidos los
  errores.
- `tests/test_api.py` prueba que **sigue respondiendo correctamente** después de
  cualquier cambio, y se puede volver a correr en cualquier momento con
  `pytest -v`.

La captura y la transcripción son una foto de un momento; las pruebas
automatizadas son lo que evita que esa foto envejezca sin que nadie se dé cuenta.

---

## Nota sobre las rutas

Las rutas que aparecen en las transcripciones y en la salida de `pytest`
corresponden a la máquina donde se ejecutaron. No afectan la reproducibilidad:
`app/main.py` calcula la ubicación del modelo a partir de la posición del propio
archivo (`BASE_DIR`), nunca de rutas absolutas ni del directorio de trabajo. Por
eso el servicio levanta igual desde la raíz del proyecto, desde `pytest` o desde
el contenedor.
