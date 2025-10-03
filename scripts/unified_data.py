import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm  # Para barras de progreso

# Ruta a los archivos parquet
parquet_pattern = "data/intermediate/parquet/*.parquet"

# Buscar todos los archivos parquet que coincidan con el patrón
parquet_files = glob.glob(parquet_pattern)
print(f"Encontrados {len(parquet_files)} archivos parquet con patrón: {parquet_pattern}")

# Inicializar contadores globales para offsets
global_entity_counter = 1
global_unique_entity_counter = 1
global_sentence_counter = 1

# Configuración para chunks (más pequeño para ahorrar memoria)
output_dir = "data/intermediate/unified_data/"
os.makedirs(output_dir, exist_ok=True)
chunk_size = 200000  # Reducido a 250k filas por chunk para menos RAM (~2-4 GiB estimado por chunk)
# Inicializar buffer para el chunk actual
current_chunk_df = pd.DataFrame()
chunk_num = 1  # Contador para nombres de archivos (global)

# Función para escribir un chunk (solo escribe, no resetea buffer) - CORREGIDA: sin parámetro chunk_num
def write_chunk(chunk_df):
    global chunk_num
    chunk_file = os.path.join(output_dir, f"unified_part_{chunk_num:03d}.parquet")
    chunk_df.to_parquet(chunk_file, index=False)
    print(f"Guardado chunk {chunk_num} en {chunk_file} ({len(chunk_df)} filas)")
    chunk_num += 1

# Procesar archivos con barra de progreso
for idx, file in enumerate(tqdm(parquet_files, desc="Procesando archivos"), start=1):
    print(f"\n[{idx}/{len(parquet_files)}] Procesando archivo: {file}")
    # Cargar el DataFrame del archivo actual
    df = pd.read_parquet(file)
    print(f"  - Filas: {len(df)} | Columnas: {len(df.columns)}")
    
    # Para sentence_id: mapear valores únicos locales a globales con offset
    local_sentences = sorted(df['sentence_id'].unique())
    sentence_map = {local_id: f"SENT_{str(global_sentence_counter + i).zfill(5)}" 
                    for i, local_id in enumerate(local_sentences)}
    df['global_sentence_id'] = df['sentence_id'].map(sentence_map)
    # Actualizar contador global
    print(f"  - Sentencias únicas locales: {len(local_sentences)} -> globales desde {global_sentence_counter} hasta {global_sentence_counter + max(len(local_sentences) - 1, 0)}")
    global_sentence_counter += len(local_sentences)
    
    # Para unique_entity_id: mapear valores únicos locales a globales con offset
    local_unique_entities = sorted(df['unique_entity_id'].unique())
    unique_entity_map = {local_id: f"UENT_{str(global_unique_entity_counter + i).zfill(5)}" 
                         for i, local_id in enumerate(local_unique_entities)}
    df['global_unique_entity_id'] = df['unique_entity_id'].map(unique_entity_map)
    # Actualizar contador global
    print(f"  - Unique entities locales: {len(local_unique_entities)} -> globales desde {global_unique_entity_counter} hasta {global_unique_entity_counter + max(len(local_unique_entities) - 1, 0)}")
    global_unique_entity_counter += len(local_unique_entities)
    
    # Para entity_id: asignar globales secuenciales para estas filas
    n_rows = len(df)
    df['global_entity_id'] = [f"ENT_{str(global_entity_counter + i).zfill(5)}" 
                              for i in range(n_rows)]
    # Actualizar contador global
    if n_rows > 0:
        start_ent = global_entity_counter
        end_ent = global_entity_counter + n_rows - 1
        print(f"  - Asignados entity_id globales a {n_rows} filas (ENT_{str(start_ent).zfill(5)} .. ENT_{str(end_ent).zfill(5)})")
    else:
        print("  - No hay filas para asignar entity_id globales")
    global_entity_counter += n_rows
    
    # Opcional: eliminar columnas originales para ahorrar memoria (~10-20% menos RAM)
    df = df.drop(columns=['entity_id', 'unique_entity_id', 'sentence_id'])
    
    # Concatenar al buffer actual del chunk
    if len(current_chunk_df) == 0:
        current_chunk_df = df.copy()
    else:
        current_chunk_df = pd.concat([current_chunk_df, df], ignore_index=True)
    print(f"  - Buffer actual tras añadir archivo: {len(current_chunk_df)} filas")
    
    # Si el buffer excede el chunk_size, escribir y resetear (CORREGIDO: preserva el restante)
    chunks_written = 0
    while len(current_chunk_df) > chunk_size:
        # Tomar las primeras chunk_size filas
        chunk_to_write = current_chunk_df.iloc[:chunk_size].copy()
        # Calcular el restante ANTES de escribir
        remaining_df = current_chunk_df.iloc[chunk_size:].reset_index(drop=True)
        # Escribir el chunk
        write_chunk(chunk_to_write)
        chunks_written += 1
        # Actualizar buffer al restante
        current_chunk_df = remaining_df
        print(f"  - Chunk {chunks_written} escrito; buffer restante: {len(current_chunk_df)} filas")

    print(f"Finalizado archivo: {file}")

# Escribir el chunk restante si hay datos
if len(current_chunk_df) > 0:
    write_chunk(current_chunk_df)

print(f"¡Unificación completada! Total de chunks: {chunk_num - 1}. Archivos en {output_dir}")
print("Tip: Usa 'htop' o 'watch free -h' en otra terminal para monitorear memoria durante la ejecución.")