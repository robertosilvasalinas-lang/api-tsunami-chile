# Imagen base liviana con la MISMA versión de Python declarada en runtime.txt
FROM python:3.14.2-slim

# Buenas prácticas: sin archivos .pyc, logs sin buffer, pip sin caché
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

# 1) Primero solo las dependencias: Docker reutiliza esta capa si el código cambia
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Luego el código y el modelo entrenado
COPY app/ ./app/
COPY model/ ./model/

# 3) Ejecutar como usuario sin privilegios (seguridad)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Chequeo de salud que usan Docker y varias nubes para saber si el servicio está vivo
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/health')" || exit 1

# Escucha en 0.0.0.0 y en el puerto que asigne la nube ($PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
