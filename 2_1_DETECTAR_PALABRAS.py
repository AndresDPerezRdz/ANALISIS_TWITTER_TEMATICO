import random
import orjson 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
import os
import json
# Ruta del archivo
ruta_archivo = './DATOS/DATOS_ENTRENAMIENTO/exported_data_muestra_20_dias.json'

# Cargar los datos
try:
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        datos_entrenar_identificar = json.load(f)
    print(f"Archivo cargado. Registros: {len(datos_entrenar_identificar)}")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    datos_entrenar_identificar = []
# Inicializar NLTK y descargar recursos necesarios
nltk.download('stopwords')

# Precompilar las expresiones regulares para mejorar el rendimiento
URL_REGEX = re.compile(r"http\S+|www\S+|https\S+")
SPECIAL_CHARS_REGEX = re.compile(r"[^a-zA-Z\s@#]")

# Definir variables globales
stop_words = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

def limpiar_texto(texto):
    """
    Limpia y procesa el texto eliminando URLs, caracteres especiales y convirtiéndolo a minúsculas.
    """
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar URLs
    texto = URL_REGEX.sub('', texto)
    # Eliminar caracteres especiales (excepto @ y #)
    texto = SPECIAL_CHARS_REGEX.sub('', texto)
    return texto

def aplicar_stemming(texto):
    """
    Aplica stemming al texto después de eliminar las stopwords.
    """
    palabras = texto.split()
    palabras_limpias = [stemmer.stem(palabra) for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras_limpias)

def procesar_datos(datos):
    """
    Procesa directamente los datos para verificar la presencia de palabras clave con stemming,
    añade un campo 'contiene_palabra' y un campo 'publicacion_steminizada'.
    """
    print("Inicio del procesamiento de texto.")
    start_time = time.time()  # Tiempo inicial

    # Lista inicial de palabras de referencia
    palabras_referencia = [
        'droga', 'narcotrafico', 'narco', 'marihuana', 'cocaina', 'heroina', 
        'cristal', 'metanfetamina', 'opio', 'fentanilo', 'cartel', 'narcotico', 
        'psicotropico', 'alucinogeno', 'traficante', 'narcomenudeo', 'incautacion', 
        'decomiso', 'contrabando', 'capo', 'anfetamina', 'lsd', 'pbc', 'hashish', 
        'dmt', 'peyote', 'mescalina', 'ketamina', 'ghb', 'popper', 'molly', 
        'extasis', 'crack', 'speedball', 'spice', 'krokodil'
    ]

    # Aplicar stemming a la lista de palabras de referencia
    palabras_referencia_stemmed = {stemmer.stem(palabra) for palabra in palabras_referencia}
    print("Palabras de referencia con stemming aplicado.")

    # Limpiar y procesar el JSON directamente
    for item in datos:
        texto = item.get('publicacion', '')
        texto_procesado = aplicar_stemming(limpiar_texto(texto))
        palabras_texto = set(texto_procesado.split())
        # Agregar los nuevos campos
        item['publicacion_steminizada'] = texto_procesado
        item['contiene_palabra'] = not palabras_referencia_stemmed.isdisjoint(palabras_texto)

    print("Procesamiento completado.")

    # Crear una subbase con las entradas que contienen palabras clave
    subbase = [item for item in datos if item['contiene_palabra']]

    # Mostrar estadísticas finales
    num_observaciones = len(subbase)
    print(f"Número de observaciones en la subbase: {num_observaciones}")
    print("Muestra de la subbase final:")
    for item in subbase[:5]:  # Mostrar solo las primeras 5 entradas
        print({
            'publicacion': item.get('publicacion', ''),
            'publicacion_steminizada': item.get('publicacion_steminizada', ''),
            'id': item.get('id', 'Sin ID'),
            'contiene_palabra': item['contiene_palabra']
        })

    # Tiempo total de ejecución
    end_time = time.time()
    print(f"Tiempo total de procesamiento: {end_time - start_time:.2f} segundos")
    return datos

# Procesar los datos cargados en la variable `datos_entrenar_identificar`
datos_entrenar_identificar = procesar_datos(datos_entrenar_identificar)

# Crear subbase con 'contiene_palabra' == True
datos_entrenar_identificar_sub = [
    item for item in datos_entrenar_identificar if item['contiene_palabra']
]

# Exportar el archivo actualizado en la misma carpeta
if datos_entrenar_identificar_sub:
    ruta_salida = os.path.join(os.path.dirname(ruta_archivo), "datos_entrenar_identificar_sub_codificado_actualizado.json")

    try:
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(datos_entrenar_identificar_sub, f, ensure_ascii=False, indent=2)
        print(f"Archivo exportado: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")