import os
import random
import json
from tqdm import tqdm  # Importar tqdm para la barra de progreso

# Directorios
input_dir = './DATOS/DATOS_JSON_LIMPIOS'
output_file = './DATOS/DATOS_ENTRENAMIENTO/exported_data_muestra_20_dias.json'

# Obtener la lista de archivos JSON en la carpeta
json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# Seleccionar hasta 20 archivos aleatorios
num_files_to_process = min(20, len(json_files))
selected_files = random.sample(json_files, num_files_to_process)

# Lista para almacenar las publicaciones extraídas
data_output = []

# Contador de ID incremental
id_counter = 1

# Procesar cada archivo JSON con tqdm
print(f"Procesando {num_files_to_process} archivos JSON...")
for json_file in tqdm(selected_files, desc="Procesando archivos", unit="archivo"):
    file_path = os.path.join(input_dir, json_file)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Cargar cada línea como JSON
                    tweet = json.loads(line.strip())
                    
                    # Extraer el texto si existe
                    if "extended_tweet" in tweet and "full_text" in tweet["extended_tweet"]:
                        data_output.append({
                            "publicacion": tweet["extended_tweet"]["full_text"],
                            "droga": 0,
                            "id": id_counter  # Asignar ID incremental
                        })
                        id_counter += 1  # Incrementar ID
                
                except json.JSONDecodeError:
                    print(f"Error al decodificar una línea en {json_file}")
    
    except Exception as e:
        print(f"Error al procesar {json_file}: {e}")

# Guardar el resultado en un nuevo archivo JSON
print("\nGuardando el archivo final...")
try:
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(data_output, out_f, ensure_ascii=False, indent=2)
    print(f"Archivo exportado correctamente: {output_file}")
except Exception as e:
    print(f"Error al guardar el archivo final: {e}")
