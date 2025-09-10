import pandas as pd
import json
import os

# Change the path to the CSV file as needed
csv_path = "./data/raw/TASTEset.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, encoding='utf-8')

# List to store the rows for the new DataFrame
rows = []
entity_counter = 0
unique_entity_dict = {}
unique_entity_counter = 0

for idx, row in df.iterrows():
    sentence = str(row['ingredients']).replace('\n', ' ').replace('\r', ' ')
    entities_str = row['ingredients_entities']
    sentence_id = idx + 1  # Unique sentence ID (1-based)
    sentence_id_str = f"SENT_{sentence_id:05d}"
    try:
        entities = json.loads(entities_str)
        for ent in entities:
            entity_counter += 1
            entity_value = ent['entity']
            # Assign a unique ID to each unique entity value
            if entity_value not in unique_entity_dict:
                unique_entity_counter += 1
                unique_entity_dict[entity_value] = unique_entity_counter
            unique_entity_id_num = unique_entity_dict[entity_value]
            entity_id_str = f"ENT_{entity_counter:05d}"
            unique_entity_id_str = f"UENT_{unique_entity_id_num:05d}"
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
                'iob_tag': iob_tag
            })
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")

entities_df = pd.DataFrame(rows)

# Save the DataFrame in the intermediate folder
output_dir = './data/intermediate/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'tasteset_entities.csv')
entities_df.to_csv(output_path, index=False, encoding='utf-8', sep = ';')
print(f"File saved at: {output_path}")