import os
import json

# Ruta donde están los archivos .json
input_dir = './'
output_dir = os.path.join(input_dir, 'txts')

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterar todos los archivos en la carpeta de entrada
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Si el json es dict o list, intenta formatearlo legiblemente
            if isinstance(data, dict) or isinstance(data, list):
                json_content = json.dumps(data, indent=4, ensure_ascii=False)
            else:
                json_content = str(data)  # fallback para otros tipos de datos
            
            # Añadir el nombre del archivo original al inicio del texto
            text_to_write = f"# Contenido del archivo original: {filename}\n\n{json_content}"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_to_write)
            print(f"Convertido: {filename} -> {output_filename}")
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

print("Conversión completada. Archivos TXT en carpeta 'txts'.")
