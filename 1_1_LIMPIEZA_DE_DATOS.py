import os
import orjson
import re
import string
import torch
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Detectar dispositivo (GPU si está disponible, de lo contrario CPU)
DEVICE = torch.device("cpu")
print(f"Usando el dispositivo: {DEVICE}")

# Ruta al directorio donde se encuentran los archivos JSON
RUTA_ENTRADA = './DATOS/DATOS_JSON'
RUTA_SALIDA = './DATOS/DATOS_JSON_LIMPIOS'

# Diccionario de estados de México
estados_mexico = {
    "distrito federal": "Ciudad de México",
    "ciudad de mexico": "Ciudad de México",
    "cdmx": "Ciudad de México",
    "mexico": "Estado de México",
    "estado de mexico": "Estado de México",
    "nuevo leon": "Nuevo León",
    "jalisco": "Jalisco",
    "veracruz de ignacio de la llave": "Veracruz",
    "veracruz": "Veracruz",
    "sonora": "Sonora",
    "quintana roo": "Quintana Roo",
    "guanajuato": "Guanajuato",
    "yucatan": "Yucatán",
    "queretaro arteaga": "Querétaro",
    "queretaro": "Querétaro",
    "tamaulipas": "Tamaulipas",
    "sinaloa": "Sinaloa",
    "baja california": "Baja California",
    "coahuila de zaragoza": "Coahuila",
    "coahuila": "Coahuila",
    "tabasco": "Tabasco",
    "morelos": "Morelos",
    "michoacan de ocampo": "Michoacán",
    "michoacan": "Michoacán",
    "hidalgo": "Hidalgo",
    "puebla": "Puebla",
    "chiapas": "Chiapas",
    "oaxaca": "Oaxaca",
    "guerrero": "Guerrero",
    "baja california sur": "Baja California Sur",
    "chihuahua": "Chihuahua",
    "nayarit": "Nayarit",
    "zacatecas": "Zacatecas",
    "san luis potosi": "San Luis Potosí",
    "colima": "Colima",
    "campeche": "Campeche",
    "tlaxcala": "Tlaxcala",
    "durango": "Durango",
    "aguascalientes": "Aguascalientes"
}

def procesar_archivo_lote(ruta_archivo, lote=100000):
    """
    Procesa un archivo JSON línea por línea, en lotes para optimizar el rendimiento,
    contando las observaciones.
    """
    print(f"Procesando archivo: {ruta_archivo}")
    total_observaciones = 0
    try:
        with open(ruta_archivo, 'rb') as archivo_json:
            buffer = []
            for i, linea in enumerate(archivo_json):
                try:
                    buffer.append(orjson.loads(linea))
                    total_observaciones += 1
                except orjson.JSONDecodeError as e:
                    print(f"Error al decodificar una línea en la línea {i + 1}: {e}")

                # Procesar en lotes
                if len(buffer) >= lote:
                    buffer = []  # Reiniciar el buffer
                    print(f"{i + 1} líneas procesadas hasta ahora...")

            # Mensaje final por archivo
            print(f"Total de líneas procesadas en {ruta_archivo}: {total_observaciones}")
    except FileNotFoundError:
        print(f"El archivo JSON especificado no existe: {ruta_archivo}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al abrir el archivo {ruta_archivo}: {e}")
    return total_observaciones

def procesar_archivos_paralelo(rutas_archivos):
    """
    Procesa múltiples archivos JSON en paralelo si hay GPU disponible,
    contando las observaciones.
    """
    total_observaciones = 0
    with ThreadPoolExecutor() as executor:
        resultados = list(executor.map(procesar_archivo_lote, rutas_archivos))
        total_observaciones = sum(resultados)
    return total_observaciones

def procesar_archivos_secuencial(rutas_archivos):
    """
    Procesa múltiples archivos JSON de manera secuencial,
    contando las observaciones.
    """
    total_observaciones = 0
    for ruta_archivo in rutas_archivos:
        total_observaciones += procesar_archivo_lote(ruta_archivo)
    return total_observaciones

def guardar_archivo(ruta_entrada, datos_filtrados, ruta_salida):
    """Guarda los datos filtrados en un archivo JSON en la ruta de salida."""
    try:
        # Extraer el nombre del archivo de la ruta de entrada
        nombre_archivo = os.path.basename(ruta_entrada)
        # Crear la ruta de salida completa
        ruta_archivo_salida = os.path.join(ruta_salida, nombre_archivo)
        
        # Guardar el archivo
        with open(ruta_archivo_salida, 'wb') as archivo:
            archivo.writelines([orjson.dumps(linea) + b'\n' for linea in datos_filtrados])
        print(f"Archivo guardado exitosamente: {ruta_archivo_salida}")
    except Exception as e:
        print(f"Error al guardar el archivo {ruta_archivo_salida}: {e}")


def procesar_y_guardar(ruta_archivo, ruta_salida):
    """
    Carga, filtra, elimina duplicados, filtra por idioma, convierte fechas,
    procesa estados, elimina campos específicos, limpia textos y guarda un archivo JSON.
    """
    try:
        print(f"Iniciando procesamiento del archivo: {ruta_archivo}")
        # Cargar datos
        try:
            print("Cargando datos...")
            with open(ruta_archivo, 'rb') as archivo_json:
                datos = [orjson.loads(linea) for linea in archivo_json]
            print(f"Datos cargados exitosamente: {len(datos)} registros.")
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            raise

        # Determinar el entorno (CPU o GPU)
        es_gpu = DEVICE.type == "cuda"
        datos_filtrados = None

        # Procesamiento en función del entorno
        if es_gpu:
            datos_filtrados = filter_data_gpu(datos)
            datos_filtrados = eliminar_duplicados_filtrar_gpu(datos_filtrados)
            datos_filtrados = convertir_fechas_gpu(datos_filtrados)
            procesar_estados_gpu(datos_filtrados)
            datos_filtrados = eliminar_campos_gpu(datos_filtrados, delete_fields)
            datos_filtrados = clean_tweets_gpu(datos_filtrados)
        else:
            datos_filtrados = filter_data_cpu(datos)
            datos_filtrados = eliminar_duplicados_filtrar_cpu(datos_filtrados)
            datos_filtrados = convertir_fechas_cpu(datos_filtrados)
            procesar_estados_cpu(datos_filtrados)
            datos_filtrados = eliminar_campos_cpu(datos_filtrados, delete_fields)
            datos_filtrados = clean_tweets_cpu(datos_filtrados)

        # Guardar el archivo sobrescribiendo en la ruta de salida
        guardar_archivo(ruta_archivo, datos_filtrados, ruta_salida)
    except Exception as e:
        print(f"Error al procesar el archivo {ruta_archivo}: {e}")

   
        
def filter_tweet(tweet):
    """Filtra un tweet basado en el campo 'place' con country_code 'MX'."""
    place = tweet.get("place")
    if place and place.get("country_code") == "MX":
        return {
            "created_at": tweet.get("created_at"),
            "id_str": tweet.get("id_str"),
            "extended_tweet": {
                "full_text": tweet.get("extended_tweet", {}).get("full_text") if tweet.get("truncated", False) else tweet.get("text")
            },
            "retweet_count": tweet.get("retweet_count"),
            "favorite_count": tweet.get("favorite_count"),
            "in_reply_to_status_id_str": tweet.get("in_reply_to_status_id_str"),
            "in_reply_to_user_id_str": tweet.get("in_reply_to_user_id_str"), 
            "user": {
                "id_str": tweet.get("user", {}).get("id_str"),
                "screen_name": tweet.get("user", {}).get("screen_name"),
                "followers_count": tweet.get("user", {}).get("followers_count"),
                "friends_count": tweet.get("user", {}).get("friends_count"),
                "verified": tweet.get("user", {}).get("verified"),
                "statuses_count": tweet.get("user", {}).get("statuses_count"),
            },
            "place": {
                "place_type": place.get("place_type"),
                "name": place.get("name"),
                "country_code": place.get("country_code"),
                "full_name": place.get("full_name"),
            },
            "lang": tweet.get("lang"),
        }
    return None

def filter_data_gpu(data):
    """Filtra los datos usando GPU."""
    # Convertir los datos a tensores de índices
    tweets_tensor = torch.tensor(range(len(data)), device=DEVICE)
    
    # Filtrar usando índices y la función map para GPU
    filtered_indices = [
        idx for idx in tweets_tensor.cpu().numpy()
        if filter_tweet(data[idx]) is not None
    ]
    
    # Construir los datos filtrados
    filtered_data = [filter_tweet(data[idx]) for idx in filtered_indices]
    return filtered_data

def filter_data_cpu(data):
    """Filtra los datos de forma secuencial en CPU."""
    filtered_data = [filter_tweet(tweet) for tweet in data if filter_tweet(tweet) is not None]
    return filtered_data

def eliminar_duplicados_filtrar_cpu(data):
    """
    Elimina duplicados y filtra por 'lang' == 'es' en CPU.
    """
    tweets_unicos = {tweet['id_str']: tweet for tweet in data if tweet['lang'] == 'es'}
    return list(tweets_unicos.values())

def eliminar_duplicados_filtrar_gpu(data):
    """
    Elimina duplicados y filtra por 'lang' == 'es' utilizando GPU.
    """
    # Convertir campos relevantes a tensores
    ids = [tweet['id_str'] for tweet in data]
    langs = [tweet['lang'] for tweet in data]

    # Crear tensores en GPU
    ids_tensor = torch.tensor([hash(id_str) for id_str in ids], device=DEVICE)
    langs_tensor = torch.tensor([1 if lang == 'es' else 0 for lang in langs], device=DEVICE)

    # Filtrar índices donde 'lang' == 'es'
    es_indices = (langs_tensor == 1).nonzero(as_tuple=True)[0]

    # Obtener ids únicos entre los filtrados
    filtered_ids = ids_tensor[es_indices]
    _, unique_indices = torch.unique(filtered_ids, return_inverse=False, return_counts=False, dim=0)

    # Reconstruir los datos filtrados y únicos
    unique_data = [data[idx] for idx in unique_indices.cpu().numpy()]
    return unique_data

# Formatos de fecha precompilados
strptime_format = '%a %b %d %H:%M:%S %z %Y'
strftime_format = '%d/%m/%Y'

def convertir_fechas_cpu(data):
    """
    Convierte las fechas en el campo 'created_at' al formato 'dd/mm/yyyy' usando CPU.
    """
    for item in data:
        created_at = datetime.strptime(item['created_at'], strptime_format)
        item['fecha'] = created_at.strftime(strftime_format)
    return data
def convertir_fechas_gpu(data):
    """
    Convierte las fechas en el campo 'created_at' al formato 'dd/mm/yyyy' usando GPU.
    """
    # Extraer los valores 'created_at' para convertir
    created_at_list = [item['created_at'] for item in data]

    # Convertir los valores a tensores (trabajando con índices en GPU)
    indices = torch.arange(len(created_at_list), device=DEVICE)

    # Usar índices para procesar en paralelo (en CPU para datetime)
    fechas_formateadas = []
    for idx in indices.cpu().numpy():
        created_at = datetime.strptime(created_at_list[idx], strptime_format)
        fechas_formateadas.append(created_at.strftime(strftime_format))
    
    # Asignar las fechas formateadas de regreso a los datos originales
    for idx, fecha in enumerate(fechas_formateadas):
        data[idx]['fecha'] = fecha

    return data

# Función para normalizar el texto
def normalizar_texto(texto):
    texto = texto.lower()
    texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
    return texto

def obtener_estado(place_full_name):
    partes = place_full_name.split(", ")
    if len(partes) > 1:
        estado = normalizar_texto(partes[-1])
        if estado in estados_mexico:
            return estados_mexico[estado]
    return "No Identificable"

def procesar_estados_cpu(data):
    """
    Procesa los estados y cuenta ocurrencias en CPU.
    """
    for item in data:
        if 'place' in item and item['place']:
            estado = obtener_estado(item['place']['full_name'])
            item['place']['estado'] = estado

    # Extraer los estados
    place_estados = [item['place']['estado'] for item in data if 'place' in item and item['place'] and item['place']['estado']]
    return Counter(place_estados)
def procesar_estados_gpu(data):
    """
    Procesa los estados y cuenta ocurrencias en GPU.
    """
    # Extraer nombres completos de lugares
    place_full_names = [item['place']['full_name'] for item in data if 'place' in item and item['place']]

    # Usar índices para procesar en paralelo (pero normalización en CPU)
    estados = []
    for name in place_full_names:
        estado = obtener_estado(name)
        estados.append(estado)

    # Asignar los estados de vuelta a los datos originales
    for idx, estado in enumerate(estados):
        data[idx]['place']['estado'] = estado

    # Contar ocurrencias
    return Counter(estados)

# Campos a eliminar
delete_fields = ['created_at', 'place.place_type', 'place.name', 'place.country_code', 'place.full_name', 'lang']

def eliminar_campos_cpu(data, fields):
    """
    Elimina los campos especificados en CPU.
    """
    for item in data:
        for field in fields:
            keys = field.split('.')
            current = item
            for key in keys[:-1]:
                if key in current:
                    current = current[key]
                else:
                    current = None
                    break
            if current and keys[-1] in current:
                del current[keys[-1]]
    return data

def eliminar_campos_gpu(data, fields):
    """
    Elimina los campos especificados en GPU.
    """
    # Usar índices para dividir y procesar en paralelo
    data_tensor = torch.tensor(range(len(data)), device=DEVICE)
    indices = data_tensor.cpu().numpy()  # GPU no puede operar sobre objetos anidados directamente

    for idx in indices:
        item = data[idx]
        for field in fields:
            keys = field.split('.')
            current = item
            for key in keys[:-1]:
                if key in current:
                    current = current[key]
                else:
                    current = None
                    break
            if current and keys[-1] in current:
                del current[keys[-1]]
    return data

# Compilar las expresiones regulares una sola vez para eficiencia
MENTION_REGEX = re.compile(r'(@\w+)')
URL_REGEX = re.compile(r'http\S+')
PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

# Función para eliminar acentos
def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

# Función optimizada para limpiar el texto del tweet
def clean_tweet(tweet):
    # Cambiar menciones por '@mencion' manteniendo el @
    tweet = MENTION_REGEX.sub('@mencion', tweet)
    
    # Sustituir enlaces por la palabra 'link'
    tweet = URL_REGEX.sub('link', tweet)
    
    # Eliminar acentos
    tweet = remove_accents(tweet)
    
    # Eliminar signos de puntuación excepto el @
    tweet = tweet.translate(str.maketrans('', '', string.punctuation.replace('@', '')))
    
    # Eliminar saltos de línea
    tweet = tweet.replace('\n', ' ')
    
    # Transformar a minúsculas y eliminar espacios adicionales en una sola operación
    return tweet.lower().strip()

def clean_tweets_cpu(data):
    """
    Limpia los tweets utilizando CPU.
    """
    for tweet in data:
        extended_tweet = tweet.get('extended_tweet')
        if extended_tweet:
            full_text = extended_tweet.get('full_text')
            if full_text:
                extended_tweet['full_text'] = clean_tweet(full_text)
    return data

def clean_tweets_gpu(data):
    """
    Limpia los tweets utilizando GPU.
    """
    # Extraer textos de los tweets
    texts = [
        tweet.get('extended_tweet', {}).get('full_text', '')
        for tweet in data
        if tweet.get('extended_tweet')
    ]

    # Usar índices para procesar en paralelo
    indices = torch.arange(len(texts), device=DEVICE)
    cleaned_texts = []

    for idx in indices.cpu().numpy():
        cleaned_texts.append(clean_tweet(texts[idx]))

    # Asignar los textos limpios de vuelta a los datos originales
    for idx, tweet in enumerate(data):
        extended_tweet = tweet.get('extended_tweet')
        if extended_tweet and idx < len(cleaned_texts):
            extended_tweet['full_text'] = cleaned_texts[idx]

    return data


# Buscar archivos JSON en el directorio de entrada
rutas_archivos_json = []
try:
    for carpeta_raiz, _, archivos in os.walk(RUTA_ENTRADA):
        for archivo in archivos:
            if archivo.lower().endswith('.json'):  # Manejo de mayúsculas/minúsculas en la extensión
                rutas_archivos_json.append(os.path.join(carpeta_raiz, archivo))
except Exception as e:
    print(f"Ocurrió un error inesperado al recorrer el directorio: {e}")

# Procesar archivos según el dispositivo disponible
if not rutas_archivos_json:
    print("No se encontraron archivos JSON en el directorio.")
else:
    total_archivos = len(rutas_archivos_json)
    print(f"Se encontró(n) {total_archivos} archivo(s) JSON.")

    if DEVICE.type == "cuda":
        print("Procesando en paralelo (GPU habilitada).")
        with ThreadPoolExecutor() as executor:
            for i, _ in enumerate(executor.map(lambda ruta: procesar_y_guardar(ruta, RUTA_SALIDA), rutas_archivos_json), 1):
                porcentaje = (i / total_archivos) * 100
                print(f"Progreso: {i}/{total_archivos} archivos procesados ({porcentaje:.2f}%)")
    else:
        print("Procesando secuencialmente (CPU).")
        for i, ruta_archivo in enumerate(rutas_archivos_json, 1):
            procesar_y_guardar(ruta_archivo, RUTA_SALIDA)
            porcentaje = (i / total_archivos) * 100
            print(f"Progreso: {i}/{total_archivos} archivos procesados ({porcentaje:.2f}%)")