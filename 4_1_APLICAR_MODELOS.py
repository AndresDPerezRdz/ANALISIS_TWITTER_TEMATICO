import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import orjson
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer

# Detectar dispositivo (GPU si está disponible, de lo contrario CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {DEVICE}")

# Cargar modelo de drogas
ruta_modelo_drogas = r"MODELOS\modelo_tinybert_drogas"
try:
    model_droga = AutoModelForSequenceClassification.from_pretrained(ruta_modelo_drogas).to(DEVICE)
    tokenizer_droga = BertTokenizer.from_pretrained(ruta_modelo_drogas)
    model_droga.eval()
    print("Modelo de drogas cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de drogas: {e}")
    exit()

# Cargar modelo de sentimientos
ruta_modelo_sentimiento = r"MODELOS\modelo_xlm_roberta_sentimientos"
try:
    model_sentimiento = AutoModelForSequenceClassification.from_pretrained(ruta_modelo_sentimiento).to(DEVICE)
    tokenizer_sentimiento = AutoTokenizer.from_pretrained(ruta_modelo_sentimiento)
    model_sentimiento.eval()
    print("Modelo de sentimientos cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de sentimientos: {e}")
    exit()

# Directorios de entrada y salida
ruta_entrada = r"DATOS\DATOS_JSON_LIMPIOS"
ruta_salida = r"DATOS\DATOS_MODELO"
os.makedirs(ruta_salida, exist_ok=True)  # Asegurar que la carpeta de salida exista

# Inicializar NLTK y descargar recursos necesarios
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

# Expresiones regulares
URL_REGEX = re.compile(r"http\S+|www\S+|https\S+")
SPECIAL_CHARS_REGEX = re.compile(r"[^a-zA-Z\s@#]")

# Lista de palabras clave relacionadas con drogas
palabras_referencia = [
    'droga', 'narcotrafico', 'narco', 'marihuana', 'cocaina', 'heroina', 
    'cristal', 'metanfetamina', 'opio', 'fentanilo', 'cartel', 'narcotico', 
    'psicotropico', 'alucinogeno', 'traficante', 'narcomenudeo', 'incautacion', 
    'decomiso', 'contrabando', 'capo', 'anfetamina', 'lsd', 'pbc', 'hashish', 
    'dmt', 'peyote', 'mescalina', 'ketamina', 'ghb', 'popper', 'molly', 
    'extasis', 'crack', 'speedball', 'spice', 'krokodil'
]

palabras_referencia_stemmed = {stemmer.stem(palabra) for palabra in palabras_referencia}

def limpiar_texto(texto):
    texto = texto.lower()
    texto = URL_REGEX.sub('', texto)
    texto = SPECIAL_CHARS_REGEX.sub('', texto)
    return texto

def aplicar_stemming(texto):
    palabras = texto.split()
    palabras_limpias = [stemmer.stem(palabra) for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras_limpias)

def clasificar_texto(datos):
    """
    Aplica el modelo de clasificación de drogas solo a los tweets relevantes.
    """
    print("Iniciando clasificación de tweets sobre drogas...")
    
    batch_texts = []
    batch_indices = []
    
    for idx, item in enumerate(datos):
        if item['contiene_palabra']:
            batch_texts.append(item['publicacion_steminizada'])
            batch_indices.append(idx)

    if not batch_texts:
        print("No hay tweets con palabras clave para clasificar.")
        return datos

    # Tokenizar y clasificar en batch
    inputs = tokenizer_droga(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        outputs = model_droga(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

    for idx, pred in zip(batch_indices, predictions):
        datos[idx]['droga'] = pred

    print("Clasificación de drogas completada.")
    return datos

def clasificar_sentimiento(datos):
    """
    Aplica el modelo de clasificación de sentimiento solo a tweets donde `droga == 1`.
    """
    print("Iniciando clasificación de sentimientos...")
    
    batch_texts = []
    batch_indices = []
    
    for idx, item in enumerate(datos):
        if item.get('droga') == 1:
            batch_texts.append(item['publicacion_steminizada'])
            batch_indices.append(idx)

    if not batch_texts:
        print("No hay tweets sobre drogas para clasificar sentimientos.")
        return datos

    # Tokenizar en batch
    inputs = tokenizer_sentimiento(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        outputs = model_sentimiento(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1).cpu().tolist()

    for idx, pred in zip(batch_indices, predictions):
        datos[idx]['sentimiento'] = pred

    print("Clasificación de sentimientos completada.")
    return datos

def procesar_datos(datos, barra_progreso):
    """
    Procesa los datos, aplica el modelo de clasificación de drogas y luego el de sentimientos.
    """
    indices = torch.arange(len(datos), device=DEVICE).cpu().numpy()
    
    for idx in indices:
        item = datos[idx]
        texto = item.get('extended_tweet', {}).get('full_text', '')
        texto_procesado = aplicar_stemming(limpiar_texto(texto))
        palabras_texto = set(texto_procesado.split())

        item['publicacion_steminizada'] = texto_procesado
        item['contiene_palabra'] = not palabras_referencia_stemmed.isdisjoint(palabras_texto)
        item['droga'] = 0 if not item['contiene_palabra'] else None

        barra_progreso.update(1)

    datos = clasificar_texto(datos)
    datos = clasificar_sentimiento(datos)
    
    return datos

# Obtener lista de archivos JSON en la carpeta de entrada
archivos_json = [f for f in os.listdir(ruta_entrada) if f.endswith(".json")]

# Contar el total de registros para la barra de progreso global
total_registros = 0
for archivo_nombre in archivos_json:
    ruta_completa_entrada = os.path.join(ruta_entrada, archivo_nombre)
    with open(ruta_completa_entrada, 'r', encoding='utf-8') as archivo:
        total_registros += sum(1 for _ in archivo)

# Barra de progreso global
with tqdm(total=total_registros, desc="Procesando registros", unit="registro") as barra_progreso:
    for archivo_nombre in archivos_json:
        ruta_completa_entrada = os.path.join(ruta_entrada, archivo_nombre)
        ruta_completa_salida = os.path.join(ruta_salida, archivo_nombre.replace(".json", "_modelo.json"))
        
        try:
            with open(ruta_completa_entrada, 'r', encoding='utf-8') as archivo:
                datos = [orjson.loads(linea) for linea in archivo]

            datos_procesados = procesar_datos(datos, barra_progreso)

            with open(ruta_completa_salida, 'wb') as archivo_salida:
                archivo_salida.write(orjson.dumps(datos_procesados))

        except Exception as e:
            print(f"Error inesperado: {e}")
