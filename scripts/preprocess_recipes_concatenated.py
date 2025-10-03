# This script preprocesses recipe data.
import pandas as pd
import os
import re
from collections import defaultdict
import ast
from tqdm import tqdm
import numpy as np

# Change the path to the CSV file as needed
csv_path_recipes = "./data/intermediate/merged_recipes_three_columns.csv"

print("Iniciando carga de datos...")
# Leer solo las columnas necesarias para ahorrar memoria
df_recipens = pd.read_csv(csv_path_recipes, encoding='utf-8', sep=';',
                         usecols=['ingredients', 'directions', 'NER'])
print(f"-> Datos cargados. Total de registros: {len(df_recipens):,}")

# =============================================================================
# VERSIÓN ULTRARRÁPIDA DE PREPROCESAMIENTO
# =============================================================================

def fast_string_to_list(s):
    """Conversión ultra rápida de string a lista"""
    if pd.isna(s) or s == '':
        return []
    s_str = str(s).strip()
    if s_str.startswith('[') and s_str.endswith(']'):
        try:
            # Método más rápido para listas simples
            return [item.strip(' "\'') for item in s_str[1:-1].split(',') if item.strip()]
        except:
            return [s_str]
    return [s_str]

print("Procesando columnas 'ingredients' y 'directions'...")

# Aplicar conversión rápida
df_recipens['ingredients'] = df_recipens['ingredients'].apply(fast_string_to_list)
df_recipens['directions'] = df_recipens['directions'].apply(fast_string_to_list)
print("-> Columnas 'ingredients' y 'directions' procesadas.")

# Crear sentence de manera más eficiente
print("Creando columna 'sentence'...")
df_recipens['sentence'] = df_recipens['ingredients'].apply(
    lambda x: ', '.join(x) if x else ''
) + ' ' + df_recipens['directions'].apply(
    lambda x: ' '.join(x) if x else ''
)
print("-> Columna 'sentence' creada.")

# Conversión rápida de NER
print("Procesando columna 'NER'...")
df_recipens['ner_entities'] = df_recipens['NER'].apply(fast_string_to_list)
print("-> Columna 'NER' procesada.")

# Filtrar solo filas con entidades NER ANTES de procesar
print("Filtrando oraciones con entidades NER...")
df_with_ner = df_recipens[df_recipens['ner_entities'].apply(len) > 0].copy()
print(f"-> Oraciones con entidades NER encontradas: {len(df_with_ner):,}")

# =============================================================================
# GUARDAR DATOS PREPROCESADOS EN CHUNKS
# =============================================================================

print("\nGuardando datos preprocesados en chunks...")
output_dir = "./data/intermediate/preprocessed_recipes_chunks/"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 100000
total_chunks = (len(df_with_ner) + chunk_size - 1) // chunk_size

chunk_files = []
for i in range(total_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df_with_ner))
    
    chunk_df = df_with_ner.iloc[start_idx:end_idx]
    chunk_filename = f"preprocessed_chunk_{i+1:03d}.parquet"
    chunk_path = os.path.join(output_dir, chunk_filename)
    
    chunk_df.to_parquet(chunk_path, index=False, engine='pyarrow')
    chunk_files.append(chunk_path)
    
    print(f"-> Chunk {i+1}/{total_chunks} guardado: {chunk_filename} ({len(chunk_df):,} registros)")

# Liberar memoria del DataFrame grande
del df_recipens, df_with_ner
print("-> Memoria liberada. Datos guardados en chunks.")

# =============================================================================
# FUNCIONES DE PROCESAMIENTO NER CON HASH PARA UNIQUE_ENTITY_ID
# =============================================================================

def fast_tokenize(sentence):
    """Tokenización ultra rápida - solo palabras"""
    return re.findall(r'\b\w+\b', sentence.lower())

def fast_find_entity(text, entity):
    """Búsqueda ultra rápida de entidad"""
    text_lower = text.lower()
    entity_lower = entity.lower()
    start = text_lower.find(entity_lower)
    if start != -1:
        return int(start), int(start + len(entity))
    else:
        return -1, -1

def fast_find_token_indices(tokens, entity):
    """Encuentra índices de tokens rápidamente"""
    entity_tokens = entity.lower().split()
    n = len(entity_tokens)

    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == entity_tokens:
            return i, i + n - 1
    return -1, -1

def fast_generate_iob(tokens, entity):
    """Genera IOB tags rápidamente"""
    entity_tokens = entity.lower().split()
    n = len(entity_tokens)

    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == entity_tokens:
            if n == 1:
                return "B-FOOD"
            else:
                return "B-FOOD " + " I-FOOD" * (n - 1)
    return ""

def generate_unique_entity_id(entity_text):
    """Genera ID único basado en hash - mismo texto = mismo ID en cualquier chunk"""
    # Usar hash absoluto y módulo para evitar números negativos y mantener rango controlado
    hash_value = abs(hash(entity_text.strip().lower())) % 10000000
    return f"UENT_{hash_value:07d}"

def process_single_chunk(chunk_path, chunk_number, total_chunks, global_entity_counter):
    """Procesa un solo chunk y guarda los resultados"""
    print(f"\nProcesando chunk {chunk_number}/{total_chunks}: {os.path.basename(chunk_path)}")
    
    # Cargar el chunk
    chunk_df = pd.read_parquet(chunk_path)
    print(f"-> Chunk cargado: {len(chunk_df):,} registros")
    
    # Procesar cada fila
    rows = []
    entity_counter = 0
    sentence_counter = 0
    
    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Chunk {chunk_number}"):
        sentence = str(row['sentence'])
        ner_entities = row['ner_entities']
        
        sentence_counter += 1
        sentence_id_str = f"SENT_{(chunk_number-1)*chunk_size + sentence_counter:07d}"

        # Precomputar una vez por oración
        tokens = fast_tokenize(sentence)
        sentence_lower = sentence.lower()

        for entity in ner_entities:
            if not entity:
                continue

            entity_text = str(entity).strip()
            if not entity_text:
                continue

            # Búsqueda rápida
            start, end = fast_find_entity(sentence_lower, entity_text)

            # Asegurar que start y end sean enteros simples, no tuplas
            start = int(start) if start != -1 else -1
            end = int(end) if end != -1 else -1

            # Encontrar tokens
            token_start, token_end = fast_find_token_indices(tokens, entity_text)

            # Generar IOB
            iob_tag_str = fast_generate_iob(tokens, entity_text) if token_start != -1 else ""

            entity_counter += 1
            global_entity_counter += 1
            
            # IDs únicos
            entity_id_str = f"ENT_{global_entity_counter:07d}"
            unique_entity_id_str = generate_unique_entity_id(entity_text)

            rows.append({
                'entity_id': entity_id_str,
                'unique_entity_id': unique_entity_id_str,
                'sentence_id': sentence_id_str,
                'entity': entity_text,
                'type': 'FOOD',
                'start': start,
                'end': end,
                'sentence': sentence,
                'iob_tag': iob_tag_str,
                'token_start': token_start,
                'token_end': token_end,
                'tokens': tokens
            })
    
    # Crear DataFrame con los resultados del chunk
    if rows:
        df_entities_chunk = pd.DataFrame(rows)
        
        # Asegurar que las columnas start y end sean del tipo correcto
        df_entities_chunk['start'] = df_entities_chunk['start'].astype(int)
        df_entities_chunk['end'] = df_entities_chunk['end'].astype(int)
        
        # Guardar resultados del chunk
        output_dir_entities = "./data/intermediate/parquet/"
        os.makedirs(output_dir_entities, exist_ok=True)
        
        output_path = os.path.join(output_dir_entities, f"ner_entities_recipes_chunk_{chunk_number:03d}.parquet")
        df_entities_chunk.to_parquet(output_path, index=False, engine='pyarrow')
        
        print(f"-> Chunk {chunk_number} procesado: {len(df_entities_chunk):,} entidades -> {output_path}")
        
        # Estadísticas del chunk
        entities_found = (df_entities_chunk['start'] != -1).sum()
        print(f"   Entidades encontradas: {entities_found:,} ({entities_found/len(df_entities_chunk)*100:.1f}%)")
        
        return len(df_entities_chunk), entities_found, global_entity_counter
    else:
        print(f"-> Chunk {chunk_number}: No se encontraron entidades")
        return 0, 0, global_entity_counter

# =============================================================================
# PROCESAR TODOS LOS CHUNKS
# =============================================================================

print(f"\n{'='*60}")
print("INICIANDO PROCESAMIENTO NER POR CHUNKS")
print(f"{'='*60}")

total_entities = 0
total_entities_found = 0
global_entity_counter = 0  # Contador global para entity_id

for i, chunk_file in enumerate(chunk_files, 1):
    entities_count, entities_found, global_entity_counter = process_single_chunk(
        chunk_file, i, len(chunk_files), global_entity_counter
    )
    total_entities += entities_count
    total_entities_found += entities_found
    
    # Liberar memoria entre chunks
    import gc
    gc.collect()

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print(f"\n{'='*60}")
print("PROCESAMIENTO COMPLETADO")
print(f"{'='*60}")
print(f"Total de chunks procesados: {len(chunk_files)}")
print(f"Total de entidades procesadas: {total_entities:,}")
print(f"Total de entidades encontradas en texto: {total_entities_found:,} ({total_entities_found/total_entities*100:.1f}%)")
print(f"\nArchivos de resultados guardados en: ./data/intermediate/parquet/")
print(f"Archivos preprocesados guardados en: ./data/intermediate/preprocessed_recipes_chunks/")

# Opcional: Combinar todos los chunks de resultados en un solo archivo
if total_entities > 0:
    combine = input("\n¿Deseas combinar todos los chunks en un solo archivo? (s/n): ")
    if combine.lower() == 's':
        print("Combinando chunks...")
        all_entities_dfs = []
        for i in range(1, len(chunk_files) + 1):
            chunk_path = f"./data/intermediate/parquet/ner_entities_recipes_chunk_{i:03d}.parquet"
            if os.path.exists(chunk_path):
                chunk_df = pd.read_parquet(chunk_path)
                all_entities_dfs.append(chunk_df)
        
        if all_entities_dfs:
            final_df = pd.concat(all_entities_dfs, ignore_index=True)
            final_output_path = "./data/intermediate/parquet/ner_entities_recipes_processed_complete.parquet"
            final_df.to_parquet(final_output_path, index=False, engine='pyarrow')
            print(f"-> Archivo combinado guardado: {final_output_path} ({len(final_df):,} entidades)")