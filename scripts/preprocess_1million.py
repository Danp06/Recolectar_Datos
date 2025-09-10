import pandas as pd
import json
import os
import glob
from collections import defaultdict

# Cargar el dataset
parquet_path = "./data/raw/HumbleIntelligence-food-ner-1-Million/train*.parquet"
parquet_files = glob.glob(parquet_path)

if parquet_files:
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True).iloc[:]
    print(f"Dataset cargado: {len(df)} ejemplos")
else:
    print("No se encontraron archivos Parquet")
    exit()

# List to store the rows for the new DataFrame
rows = []
entity_counter = 0
unique_entity_dict = {}
unique_entity_counter = 0
sentence_counter = 0

def find_all_occurrences(text, substring):
    """Encuentra todas las ocurrencias de un substring en el texto"""
    start = 0
    occurrences = []
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        end = start + len(substring)
        occurrences.append((start, end))
        start = end
    return occurrences

for idx, row in df.iterrows():
    sentence = str(row['sentence']).lower()
    tokens = row['nltk_tokens']
    iob_tags = row['iob_tags']
    
    sentence_counter += 1
    sentence_id_str = f"SENT_{sentence_counter:05d}"
    
    # Primero: recolectar todas las entidades con sus posiciones de token
    entities_info = []
    current_entity = []
    current_type = None
    token_start = None
    
    for i, (token, tag) in enumerate(zip(tokens, iob_tags)):
        if tag.startswith('B-'):
            # Guardar entidad anterior si existe
            if current_entity:
                entity_text = ' '.join(current_entity)
                entities_info.append({
                    'text': entity_text,
                    'type': current_type,
                    'token_start': token_start,
                    'token_end': i - 1
                })
            
            # Iniciar nueva entidad
            current_entity = [token]
            current_type = tag[2:]
            token_start = i
            
        elif tag.startswith('I-') and current_entity is not None and current_type == tag[2:]:
            # Continuar entidad actual
            current_entity.append(token)
            
        elif (tag == 'O' or tag.startswith('B-')) and current_entity:
            # Finalizar entidad actual
            entity_text = ' '.join(current_entity)
            entities_info.append({
                'text': entity_text,
                'type': current_type,
                'token_start': token_start,
                'token_end': i - 1
            })
            
            # Resetear
            current_entity = []
            current_type = None
            token_start = None
            
            # Si es un nuevo B-, procesarlo
            if tag.startswith('B-'):
                current_entity = [token]
                current_type = tag[2:]
                token_start = i
    
    # Procesar la última entidad si existe
    if current_entity:
        entity_text = ' '.join(current_entity)
        entities_info.append({
            'text': entity_text,
            'type': current_type,
            'token_start': token_start,
            'token_end': len(tokens) - 1
        })
    
    # Track de ocurrencias ya usadas para cada tipo de entidad
    used_occurrences = defaultdict(list)
    
    # Segundo: encontrar las posiciones reales en el texto
    for entity_info in entities_info:
        entity_text = entity_info['text']
        occurrences = find_all_occurrences(sentence, entity_text)
        
        if not occurrences:
            # Si no se encuentra, usar una aproximación
            start = -1
            end = -1
        else:
            # Encontrar la primera ocurrencia que no haya sido usada para esta entidad
            for occ in occurrences:
                if occ not in used_occurrences[entity_text]:
                    start, end = occ
                    used_occurrences[entity_text].append(occ)
                    break
            else:
                # Si todas las ocurrencias ya fueron usadas, usar la primera
                start, end = occurrences[0]
        
        entity_counter += 1
        
        # Asignar ID único a cada valor de entidad único
        if entity_text not in unique_entity_dict:
            unique_entity_counter += 1
            unique_entity_dict[entity_text] = unique_entity_counter
        
        unique_entity_id_num = unique_entity_dict[entity_text]
        entity_id_str = f"ENT_{entity_counter:05d}"
        unique_entity_id_str = f"UENT_{unique_entity_id_num:05d}"
        
        # Crear IOB tag completo
        token_count = entity_info['token_end'] - entity_info['token_start'] + 1
        if token_count == 1:
            iob_tag_str = f"B-{entity_info['type']}"
        else:
            iob_tags_list = [f"B-{entity_info['type']}"] + [f"I-{entity_info['type']}"] * (token_count - 1)
            iob_tag_str = ' '.join(iob_tags_list)
        
        rows.append({
            'entity_id': entity_id_str,
            'unique_entity_id': unique_entity_id_str,
            'sentence_id': sentence_id_str,
            'entity': entity_text,
            'type': entity_info['type'],
            'start': start,
            'end': end,
            'sentence': sentence,
            'iob_tag': iob_tag_str,
            'token_start': entity_info['token_start'],
            'token_end': entity_info['token_end']
        })

# Crear DataFrame
entities_df = pd.DataFrame(rows)

# Verificar y corregir duplicados exactos
print("Buscando duplicados exactos...")
duplicate_mask = entities_df.duplicated(['sentence_id', 'entity', 'start', 'end'], keep=False)
duplicates = entities_df[duplicate_mask]

if len(duplicates) > 0:
    print(f"Encontrados {len(duplicates)} registros duplicados, eliminando...")
    # Mantener solo la primera ocurrencia de cada duplicado
    entities_df = entities_df.drop_duplicates(['sentence_id', 'entity', 'start', 'end'])
    print(f"Dataset después de eliminar duplicados: {len(entities_df)} registros")

# Estadísticas
print(f"\nEstadísticas del dataset procesado:")
print(f"Total de oraciones: {sentence_counter}")
print(f"Total de entidades: {len(entities_df)}")
print(f"Entidades únicas: {len(unique_entity_dict)}")

# Guardar el DataFrame
output_dir = './data/intermediate/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'humble_intelligence_1million_entities.csv')
entities_df.to_csv(output_path, index=False, encoding='utf-8', sep=';')
print(f"\nArchivo guardado en: {output_path}")