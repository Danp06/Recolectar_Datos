import pandas as pd
import json
import os
from tqdm import tqdm

# Change the path to the CSV file as needed
csv_path = "./data/raw/TASTEset.csv"

print("Iniciando carga de datos...")
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, encoding='utf-8')
print(f"-> Datos cargados. Total de registros: {len(df):,}")

# List to store the rows for the new DataFrame
rows = []
entity_counter = 0
unique_entity_dict = {}
unique_entity_counter = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando oraciones"): # Añadir tqdm aquí
    # Obtener los tokens de la oración (sentence)
    sentence = str(row['ingredients']).replace('\n', ' ').replace('\r', ' ')
    tokens = sentence.split()
    # Calcular spans de caracteres para cada token
    token_spans = []  # lista de tuplas (start_char, end_char)
    cursor = 0
    for token in tokens:
        start_pos = sentence.find(token, cursor)
        if start_pos == -1:
            # No encontrado desde cursor; intentar búsqueda global como fallback
            start_pos = sentence.find(token)
        end_pos = start_pos + len(token)
        token_spans.append((start_pos, end_pos))
        cursor = end_pos
    entities_str = row['ingredients_entities']
    sentence_id = idx + 1  # Unique sentence ID (1-based)
    sentence_id_str = f"SENT_{sentence_id:05d}"
    try:
        entities = json.loads(entities_str)
        for ent in tqdm(entities, desc=f"Extrayendo entidades para SENT_{sentence_id_str}", leave=False):
            entity_counter += 1
            entity_value = ent['entity']
            # Assign a unique ID to each unique entity value
            if entity_value not in unique_entity_dict:
                unique_entity_counter += 1
                unique_entity_dict[entity_value] = unique_entity_counter
            unique_entity_id_num = unique_entity_dict[entity_value]
            entity_id_str = f"ENT_{entity_counter:05d}"
            unique_entity_id_str = f"UENT_{unique_entity_id_num:05d}"
            # Mapear a índices de tokens
            ent_start = ent['start']
            ent_end = ent['end']
            token_start_idx = -1
            token_end_idx = -1
            for i, (tok_start, tok_end) in enumerate(token_spans):
                # solapamiento si tok_end > ent_start y tok_start < ent_end
                if tok_end > ent_start and tok_start < ent_end:
                    token_start_idx = i
                    break
            if token_start_idx != -1:
                for j in range(len(token_spans) - 1, -1, -1):
                    tok_start, tok_end = token_spans[j]
                    if tok_end > ent_start and tok_start < ent_end:
                        token_end_idx = j
                        break
            # IOB tag logic
            entity_words = str(entity_value).split()
            if len(entity_words) == 1:
                iob_tag = f"B-{ent['type']}"
            else:
                iob_tag = ' '.join([f"B-{ent['type']}"] + [f"I-{ent['type']}" for _ in entity_words[1:]])
            rows.append({
                'entity_id': entity_id_str,                 # e.g. ENT_00001
                'unique_entity_id': unique_entity_id_str,   # e.g. UENT_00001
                'sentence_id': sentence_id_str,             # e.g. SENT_00001
                'entity': entity_value,
                'type': ent['type'],
                'start': ent['start'],
                'end': ent['end'],
                'sentence': sentence,
                'iob_tag': iob_tag,
                'token_start': token_start_idx,
                'token_end': token_end_idx,
                'tokens': tokens
            })
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")

entities_df = pd.DataFrame(rows)

print("\nEstadísticas del dataset procesado:")
print(f"-> Total de entidades extraídas: {len(entities_df):,}")
print(f"-> Entidades únicas identificadas: {len(unique_entity_dict):,}")

# Save the DataFrame in the intermediate folder
output_dir = './data/intermediate/parquet/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'tasteset_entities.parquet')
entities_df.to_parquet(output_path, index=False, engine='pyarrow')
print(f"\nArchivo Parquet guardado exitosamente en: {output_path}")