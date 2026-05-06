# %%
# =============================================================================
# IMPORTS Y CONFIGURACIÓN
# =============================================================================
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import gc
import psutil
import os
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración para plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)

# Paths
RAW_DIR = Path("../data/raw/cluvi/")
OUTPUT_DIR = "../data/process/"  # Adaptado si es necesario
REPORT_DIR = Path("../reports/eda_cluvi/")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EDA COMPLETO Y MEJORADO - DATASET CLUVI (MENÚS Y CATEGORÍAS)")
print("="*80)

# %%
# Paths
OUTPUT_DIR = "../data/process/"
sentences_path = f"{OUTPUT_DIR}sentences_final.parquet"
entities_path = f"{OUTPUT_DIR}entities.parquet"
main_path = f"{OUTPUT_DIR}main_cleaned.parquet"
REPORT_DIR = Path("../reports/eda/")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Análisis de cada uno de los DataFrames usando Polars
print("\n======= Análisis rápido de los DataFrames con polars =======\n")

# Sentences DataFrame
try:
    sentences_df = pl.read_parquet(sentences_path)
    print(f"Sentences DF: {sentences_df.shape[0]:,} filas, {sentences_df.shape[1]} columnas")
    print(sentences_df.head(3))
    print(sentences_df.describe())
except Exception as e:
    print(f"No se pudo leer {sentences_path}: {e}")

print("\n-------------------------------------\n")


# %%
# Mostrar toda la información (no truncada) del registro con id SENT_22548,
# especialmente para columnas tipo lista o texto largo.
sent_id = "SENT_22548"
if 'sentences_df' in locals():
    info_sent_22548 = sentences_df.filter(pl.col("sentence_id") == sent_id)
    if info_sent_22548.height == 0:
        print(f"No se encontró información para el id {sent_id}.")
    else:
        print(f"Información para el id {sent_id}:")
        # Imprimir cada columna por separado para evitar truncamientos
        for row in info_sent_22548.to_dicts():
            for key, value in row.items():
                print(f"{key}: {value}\n{'-'*50}")
else:
    print("El DataFrame 'sentences_df' no está cargado.")


# %%

# Entities DataFrame
try:
    entities_df = pl.read_parquet(entities_path)
    print(f"Entities DF: {entities_df.shape[0]:,} filas, {entities_df.shape[1]} columnas")
    print(entities_df.head(3))
    print(entities_df.describe())
except Exception as e:
    print(f"No se pudo leer {entities_path}: {e}")

print("\n-------------------------------------\n")


# %%

# Main DataFrame
try:
    main_df = pl.read_parquet(main_path)
    print(f"Main DF: {main_df.shape[0]:,} filas, {main_df.shape[1]} columnas")
    print(main_df.head(3))
    print(main_df.describe())
except Exception as e:
    print(f"No se pudo leer {main_path}: {e}")



# %%
# Mostrar toda la información (no truncada) del registro con id SENT_22548,
# especialmente para columnas tipo lista o texto largo.
sent_id = "SENT_22548"
if 'main_df' in locals():
    info_sent_22548 = main_df.filter(pl.col("sentence_id") == sent_id)
    if info_sent_22548.height == 0:
        print(f"No se encontró información para el id {sent_id}.")
    else:
        print(f"Información para el id {sent_id}:")
        # Imprimir cada columna por separado para evitar truncamientos
        for row in info_sent_22548.to_dicts():
            for key, value in row.items():
                print(f"{key}: {value}\n{'-'*50}")
else:
    print("El DataFrame 'main_df' no está cargado.")


# %%



