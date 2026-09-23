# Datos

**Fuente:** U.S. Geological Survey (USGS), *Earthquake Catalog* (ComCat), consultado mediante la
API pública FDSN Event: https://earthquake.usgs.gov/fdsnws/event/1/

**Cita sugerida:** U.S. Geological Survey, Earthquake Hazards Program (2017). *Advanced National
Seismic System (ANSS) Comprehensive Catalog of Earthquake Events and Products.*
https://doi.org/10.5066/F7MS3QZH

**Volumen obtenido:** 1.584 sismos, de los cuales 22 (1,39 %) tienen el flag de tsunami.

**Filtro usado** (definido en `train.py`, variable `USGS_PARAMS`):
sismos de magnitud ≥ 5.0 entre el 01-01-2010 y el 01-01-2026, en la caja
latitud −56 a −17 y longitud −80 a −66 (Chile y alrededores).

**Cómo obtenerlos:** no hay que hacer nada manual. `python train.py` los descarga y guarda una
copia en `data/sismos_chile_usgs.csv`. Ese CSV está en `.gitignore` (no se sube al repositorio);
si se borra, `train.py` lo vuelve a descargar.

| Columna | Tipo | Descripción |
|---|---|---|
| magnitude | numérica | Magnitud del sismo |
| depth | numérica | Profundidad en km |
| lat, lon | numéricas | Coordenadas del epicentro |
| mag_type | **categórica** | Método con que se calculó la magnitud (mww, mb, mwr, ...) |
| tsunami | objetivo (0/1) | Flag de USGS: 1 = evento grande en región oceánica con información del sistema de alerta de tsunamis de NOAA |

> **Nota 1:** según USGS, el flag `tsunami` indica que el evento cumple condiciones para que se emita
> información de tsunami; **no** confirma que haya ocurrido un tsunami.
>
> **Nota 2 — por qué la ventana parte en 2010:** al explorar el catálogo desde 1990 comprobamos que
> todos los eventos anteriores a 2010 tenían el flag en 0, incluso sismos de gran magnitud. USGS no
> registra ese indicador para los eventos antiguos, así que incluirlos habría agregado más de mil
> etiquetas falsamente negativas. `train.py` imprime el conteo de casos con alerta por quinquenio,
> que es la evidencia de esta decisión.
