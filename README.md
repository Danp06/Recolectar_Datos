# Recolectar_Datos

Recolectar_Datos es un proyecto para la recolección, limpieza y preprocesamiento de datos de recetas de cocina, enfocado en la extracción y normalización de ingredientes y entidades alimentarias a partir de datasets públicos.

## Estructura del proyecto

- `data/raw/`         : Datasets originales sin procesar (ej: TASTEset, RecipeNLG, 3A2M, HumbleIntelligence-food-ner).
- `data/processed/`   : Datos procesados/listos para análisis.
- `utilities/`        : Utilidades de preprocesamiento de texto.
- `scripts/`          : Scripts para procesamiento y análisis.
- `notebooks/`        : Notebooks para experimentos y análisis exploratorio.
- `models/`           : Modelos entrenados o en desarrollo.
- `logs/`             : Logs de ejecución.
- `requirements.txt`  : Dependencias del proyecto.
- `setup.py`          : Instalación como paquete Python.

## Dependencias principales

Para ver la lista completa y las versiones exactas, consulta el archivo [`requirements.txt`](requirements.txt).

## Creación y activación del entorno

Primero, debes crear un entorno virtual (por ejemplo, usando `python -m venv venv` o `python3 -m venv venv`). Una vez creado, puedes usar el script `activate_env.sh` para activar automáticamente un entorno virtual detectado en la carpeta o subcarpetas.

## Contacto

Para dudas o sugerencias, contacta a Daniel o abre un issue en el repositorio.