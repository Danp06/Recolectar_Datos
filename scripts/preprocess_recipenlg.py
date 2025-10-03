import pandas as pd
import json
import os

# Change the path to the CSV file as needed
csv_path_recipenlg = "./data/raw/RecipeNLG/RecipeNLG_dataset.csv"
csv_path_3a2m = "./data/raw/3A2M/3A2M.csv"

# Read the CSV file into a pandas DataFrame
df_recipenlg = pd.read_csv(csv_path_recipenlg, encoding='utf-8')
df_recipenlg = df_recipenlg.loc[:, ~df_recipenlg.columns.str.startswith('Unnamed: 0_')]
df_3a2m = pd.read_csv(csv_path_3a2m, encoding='utf-8')
df_3a2m = df_3a2m.loc[:, ~df_3a2m.columns.str.startswith('Unnamed: 0_')]

# Mostrar más filas y columnas para ver el DataFrame más grande
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)

print("RecipeNLG DataFrame Head:")
print(df_recipenlg.head(10))
print("\n3A2M DataFrame Head:")
print(df_3a2m.head(10))

# Verificar columnas disponibles
print("\nColumnas de RecipeNLG:", df_recipenlg.columns.tolist())
print("Columnas de 3A2M:", df_3a2m.columns.tolist())

# Validar que ambas tienen las columnas necesarias
if all(col in df_recipenlg.columns for col in ['title', 'directions', 'NER']) and \
   all(col in df_3a2m.columns for col in ['title', 'directions', 'NER']):
    
    print("\n✓ Todas las columnas 'title', 'directions' y 'NER' existen en ambos DataFrames")
    
    # Unir los DataFrames por las tres columnas
    df_merged = pd.merge(df_recipenlg, df_3a2m, 
                         on=['title', 'directions', 'NER'], 
                         how='outer',
                         suffixes=('_recipenlg', '_3a2m'),
                         indicator=True)  # Añadir columna para ver origen de cada registro
    
    # Eliminar columnas de índice no deseadas que pueden aparecer después de la unión
    df_merged = df_merged.loc[:, ~df_merged.columns.str.startswith('Unnamed: 0_')]
    
    # Mostrar información sobre el DataFrame resultante
    print(f"\n=== RESULTADO DE LA UNIÓN ===")
    print(f"Tamaño del DataFrame unido: {df_merged.shape}")
    print(f"Recetas únicas: {df_merged[['title', 'directions', 'NER']].drop_duplicates().shape[0]}")
    
    # Estadísticas de la unión usando la columna _merge
    merge_counts = df_merged['_merge'].value_counts()
    print(f"\n--- Estadísticas de la unión ---")
    print(f"Total de registros: {len(df_merged)}")
    print(f"Coincidencias exactas en ambas bases: {merge_counts.get('both', 0)} ({merge_counts.get('both', 0)/len(df_merged):.1%})")
    print(f"Solo en RecipeNLG: {merge_counts.get('left_only', 0)} ({merge_counts.get('left_only', 0)/len(df_merged):.1%})")
    print(f"Solo en 3A2M: {merge_counts.get('right_only', 0)} ({merge_counts.get('right_only', 0)/len(df_merged):.1%})")
    
    # Mostrar primeras filas del resultado
    print(f"\n--- Primeras 10 filas del DataFrame unido ---")
    print(df_merged.head(10))
    
    # Opcional: guardar el DataFrame unido
    output_path = "./data/intermediate/merged_recipes_three_columns.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding='utf-8', sep = ';')
    print(f"\n✓ DataFrame guardado como '{output_path}'")
    
    # Mostrar información sobre columnas resultantes
    print(f"\n--- Columnas del DataFrame unido ---")
    for i, col in enumerate(df_merged.columns.tolist(), 1):
        print(f"{i:2d}. {col}")
    
else:
    print("\n✗ Error: No se encontraron todas las columnas necesarias en ambos DataFrames")
    missing_cols_recipenlg = [col for col in ['title', 'directions', 'NER'] if col not in df_recipenlg.columns]
    missing_cols_3a2m = [col for col in ['title', 'directions', 'NER'] if col not in df_3a2m.columns]
    
    if missing_cols_recipenlg:
        print(f"Faltan en RecipeNLG: {missing_cols_recipenlg}")
    if missing_cols_3a2m:
        print(f"Faltan en 3A2M: {missing_cols_3a2m}")