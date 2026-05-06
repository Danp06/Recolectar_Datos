# %%
import os
import glob
import shutil
import gc  # Para limpieza de memoria
from pathlib import Path

import polars as pl

# Configuración global de Polars para chunks pequeños en streaming
pl.Config.set_streaming_chunk_size(25_000)  # Bajado para menos RAM

# Directorios de trabajo
DATA_INTERMEDIATE = "../data/intermediate/unified_data/"
TEMP_DIR = Path("../data/temp/")
OUTPUT_DIR = Path("../data/process/")
OUTPUT_DIR.mkdir(exist_ok=True)


# %%
# Limpia temps previos si existen
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

# Función para procesar oraciones: crea partes temporales por archivo CON unique local
def process_sentences_to_parts():
    parquet_files = sorted(glob.glob(f"{DATA_INTERMEDIATE}*.parquet"))
    print(f"Encontrados {len(parquet_files)} archivos.")
    
    TEMP_DIR.mkdir(exist_ok=True)
    temp_file = TEMP_DIR / "sentences_parts"
    
    part_num = 0
    total_rows = 0
    
    for file in parquet_files:
        print(f"Procesando {Path(file).name} ({part_num+1}/{len(parquet_files)})...")
        df_batch = (
            pl.scan_parquet(file)
            .select([
                pl.col("global_sentence_id").alias("sentence_id"),
                pl.col("sentence"),
                pl.col("tokens"),
            ])
            .unique(subset=["sentence_id"], maintain_order=False)  # Unique LOCAL: reduce aquí!
            .collect(engine="streaming")
        )
        
        rows_in_batch = len(df_batch)
        total_rows += rows_in_batch
        print(f"  - Filas después de unique local: {rows_in_batch}")
        
        # Escribe parte numerada
        part_path = temp_file.with_suffix(f".part{part_num:03d}.parquet")
        df_batch.write_parquet(part_path, compression="snappy", use_pyarrow=False)
        del df_batch  # Libera RAM
        gc.collect()  # Fuerza limpieza después de cada archivo
        
        part_num += 1
    
    print(f"¡Listo! {part_num} partes creadas, total filas REDUCIDAS aprox: {total_rows}")
    return part_num

# Función para unir partes SIN ops globales – solo sink raw (streaming puro)
def union_raw_sentences():
    parts = sorted(glob.glob(str(TEMP_DIR / "sentences_parts.part*.parquet")))
    print(f"Uniendo {len(parts)} partes ya reducidas en raw...")
    
    sentences_lazy = pl.scan_parquet(parts)
    
    # Sink simple sin unique/sort
    raw_output = OUTPUT_DIR / "sentences_raw.parquet"
    sentences_lazy.sink_parquet(raw_output, engine="streaming", compression="snappy")
    
    # Verifica conteo raw (debería ser ~3.4M)
    raw_count = pl.scan_parquet(raw_output).select(pl.len()).collect()
    print(f"Raw unido: {raw_count} filas")
    
    return raw_output

# Función para unique global + sort en raw (ahora súper pequeño, collect directo)
def finalize_sentences(raw_path):
    print("Aplicando unique GLOBAL y sort al raw reducido...")
    
    sentences_final_lazy = (
        pl.scan_parquet(raw_path)
    )
    
    # Collect: con ~3M filas post-unique local, <1GB RAM fácil
    sentences_final = sentences_final_lazy.collect()
    
    output_path = OUTPUT_DIR / "sentences_prefinal.parquet"
    sentences_final.write_parquet(output_path, compression="snappy")
    
    print(f"Final: {len(sentences_final)} filas únicas TOTALES ordenadas en {output_path}")
    return output_path

# Función para procesar entidades (mismo enfoque si OOM: unique local primero, pero por ahora directo)
def process_entities():
    df = pl.scan_parquet(f"{DATA_INTERMEDIATE}*.parquet")
    
    print("Esquema de los datos:")
    print(df.schema)
    
    entities_lazy = (
        df
        .select([
            pl.col("global_unique_entity_id").alias("entity_id"),
            pl.col("entity"),
            pl.col("type").alias("type_entity"),
            pl.col("iob_tag"),
        ])
        .unique(subset=["entity_id"])
        .sort("entity_id")
    )
    
    entities_df = entities_lazy.collect()
    output_entities_path = OUTPUT_DIR / "entities.parquet"
    entities_df.write_parquet(output_entities_path, compression="snappy")
    
    print(f"Entidades procesadas: {len(entities_df)} filas guardadas en {output_entities_path}")
    return entities_df

# Función para limpiar temps y raw
def cleanup_temp():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("Temps limpiados.")
    
    raw_path = OUTPUT_DIR / "sentences_raw.parquet"
    if raw_path.exists():
        os.remove(raw_path)
        print("Raw intermedio eliminado.")

# %%

# Procesa oraciones
process_sentences_to_parts()
raw_path = union_raw_sentences()
sentences_path = finalize_sentences(raw_path)
cleanup_temp()


# %%
# Cargamos el archivo parquet, seleccionamos las columnas, eliminamos duplicados por 'sentence_id' y ordenamos
sentences_df = (
    pl.scan_parquet("../data/process/sentences_prefinal.parquet")
    .unique(subset=["sentence_id"])
    .sort("sentence_id")
    .collect(engine="streaming")
)


output_entities_path = OUTPUT_DIR / "sentences_final.parquet"
sentences_df.write_parquet(output_entities_path, compression="snappy")

# %%

# Procesa entidades
entities_df = process_entities()

print("\n¡Todo completado!")
print(f"- Oraciones finales: {sentences_path}")
print(f"- Entidades: {OUTPUT_DIR / 'entities.parquet'}")

# %%
# Función actualizada para limpiar la tabla principal: rename + reorder (sin drop ni join)
def clean_main_table():
    # Escanea los originales
    df_main = pl.scan_parquet(f"{DATA_INTERMEDIATE}*.parquet")
    
    # Muestra esquema para confirmar columnas
    print("Esquema original:")
    print(df_main.schema)
    
    # Rename y reorder: renombra los globals a los nombres limpios, y ordena
    main_cleaned_lazy = df_main.select([
        pl.col("global_entity_id").alias("entity_local_id"),   # Renombra a entity_local_id (ID de la tabla)
        pl.col("global_sentence_id").alias("sentence_id"),     # Renombra a sentence_id
        pl.col("global_unique_entity_id").alias("entity_id"),  # Renombra a entity_id
        pl.col("start").alias("char_start"),                   # Renombra a char_start (inicio char en sentence)
        pl.col("end").alias("char_end"),                       # Renombra a char_end (fin char en sentence)
        pl.col("token_start"),
        pl.col("token_end"),
    ])
    
    # Ver plan
    print(main_cleaned_lazy.explain(engine="streaming", optimized=True))
    
    # Sink directo
    cleaned_path = OUTPUT_DIR / "main_cleaned.parquet"
    main_cleaned_lazy.sink_parquet(cleaned_path, engine="streaming", compression="snappy")
    
    # Verifica
    final_count = pl.scan_parquet(cleaned_path).select(pl.len()).collect()
    print(f"Tabla principal limpiada: {final_count} filas en {cleaned_path}")
    print("Esquema final:")
    print(pl.scan_parquet(cleaned_path).schema)
    
    return cleaned_path

# Llama (después de sentences y entities, pero como ya están renombrados, no necesitas join)
main_cleaned = clean_main_table()


main_cleaned = clean_main_table()

print("\n¡Limpieza completada!")
print(f"- Tabla principal sin IDs: {main_cleaned}")


