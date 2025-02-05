import json

# Ruta del archivo de entrada y salida
input_path = './DATOS/DATOS_ENTRENAMIENTO/datos_entrenar_identificar_sub_codificado_actualizado.json'
output_path = './DATOS/DATOS_ENTRENAMIENTO/datos_codificados_sentimientos.json'


# Cargar los datos del archivo JSON
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Filtrar solo los registros donde "droga" sea igual a 1
filtered_data = [record for record in data if record.get("droga") == 1]

# Agregar el campo "sentimiento" con valor 0 a cada registro
for record in filtered_data:
    record["sentimiento"] = 0

# Guardar los datos modificados en un nuevo archivo JSON
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print("Archivo procesado y guardado en:", output_path)
