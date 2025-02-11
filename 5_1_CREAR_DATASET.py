import os
import glob
import pandas as pd
import orjson
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Verificar si hay una GPU disponible
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dispositivo.type == "cuda":
    print("Se usará la GPU:", torch.cuda.get_device_name(0))
else:
    print("No se detectó GPU, se usará la CPU.")

# Directorio que contiene los archivos JSON
directorio_json = r"DATOS\DATOS_MODELO"
lista_archivos = glob.glob(os.path.join(directorio_json, "*.json"))
print(f"Se encontraron {len(lista_archivos)} archivos JSON en {directorio_json}")

# Estructuras para acumular resultados de cada archivo
lista_df = []       # Para concatenar DataFrames y crear el dataset general
lista_drogas = []   # Para acumular registros filtrados donde 'droga' == 1
usuarios_global = defaultdict(lambda: {
    'followers_count': 0,
    'friends_count': 0,
    'statuses_count': 0,
    'total_publicaciones': 0,
    'interacciones_totales': 0,
    'respuestas_recibidas': 0,
    'screen_name': '',
    'sentimientos': {'negativa': 0, 'neutral': 0, 'positiva': 0}
})

# Procesar cada archivo individualmente
for archivo in lista_archivos:
    print(f"\nProcesando archivo: {archivo}")
    with open(archivo, 'rb') as f:
        datos = orjson.loads(f.read())
    print(f"Número de registros en el archivo: {len(datos)}")
    
    # Convertir los datos en DataFrame y realizar preprocesamiento
    df = pd.DataFrame(datos)
    # Procesamiento inicial (para ambos dispositivos, simplificado en CPU)
    df['estado'] = df['place'].apply(lambda x: x.get('estado', None))
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
    df['anio'] = df['fecha'].dt.year

    # Acumular DataFrame para el dataset general
    lista_df.append(df)

    # Filtrar registros donde 'droga' == 1 y acumularlos
    df_droga = df[df['droga'] == 1].copy()
    lista_drogas.append(df_droga)
    
    # Actualizar datos de usuarios (actores) con registros filtrados
    for registro in df_droga.to_dict(orient='records'):
        user_id = registro['user']['id_str']
        usuario = usuarios_global[user_id]
        usuario['followers_count'] = registro['user'].get('followers_count', 0)
        usuario['friends_count'] = registro['user'].get('friends_count', 0)
        usuario['statuses_count'] = registro['user'].get('statuses_count', 0)
        usuario['total_publicaciones'] += 1
        usuario['interacciones_totales'] += registro.get('retweet_count', 0) + registro.get('favorite_count', 0)
        usuario['screen_name'] = registro['user'].get('screen_name', '')
        sentimiento = registro.get('sentimiento')
        if sentimiento == 0:
            usuario['sentimientos']['negativa'] += 1
        elif sentimiento == 1:
            usuario['sentimientos']['neutral'] += 1
        elif sentimiento == 2:
            usuario['sentimientos']['positiva'] += 1

    # Liberar variables temporales
    del datos, df, df_droga

print("\nProcesamiento de todos los archivos completado.")

# Combinar todos los DataFrames para el dataset general
df_total = pd.concat(lista_df, ignore_index=True)
print(f"Número total de registros combinados: {len(df_total)}")

# Crear y exportar el dataset general agrupado por 'estado'
if dispositivo.type != "cuda":
    dataset_general = df_total.groupby('estado').agg(
        total_observaciones=('id_str', 'count'),
        droga_1=('droga', 'sum'),
        sentimiento_0=('sentimiento', lambda x: (x == 0).sum()),
        sentimiento_1=('sentimiento', lambda x: (x == 1).sum()),
        sentimiento_2=('sentimiento', lambda x: (x == 2).sum())
    ).reset_index()
    ruta_csv_general = "dataset_general.csv"
    dataset_general.to_csv(ruta_csv_general, index=False, encoding='latin-1')
    print(f"Archivo CSV general exportado en: {ruta_csv_general}")

    # Crear y exportar datasets por año (2019 a 2022)
    for anio in range(2019, 2023):
        dataset_anio = df_total[df_total['anio'] == anio].groupby('estado').agg(
            total_observaciones=('id_str', 'count'),
            droga_1=('droga', 'sum'),
            sentimiento_0=('sentimiento', lambda x: (x == 0).sum()),
            sentimiento_1=('sentimiento', lambda x: (x == 1).sum()),
            sentimiento_2=('sentimiento', lambda x: (x == 2).sum())
        ).reset_index()
        ruta_csv_anio = f"dataset_{anio}.csv"
        dataset_anio.to_csv(ruta_csv_anio, index=False, encoding='latin-1')
        print(f"Archivo CSV para el año {anio} exportado en: {ruta_csv_anio}")
else:
    print("Procesamiento en GPU no implementado para la agregación de datasets generales.")

# Análisis de followers a partir de los registros filtrados de 'droga'
df_drogas_total = pd.concat(lista_drogas, ignore_index=True)
followers_count = [registro['user']['followers_count'] for registro in df_drogas_total.to_dict(orient='records')]
if followers_count:
    if dispositivo.type == "cuda":
        followers_tensor = torch.tensor(followers_count, dtype=torch.float32, device=dispositivo)
        avg_followers = followers_tensor.mean().item()
        sorted_followers = torch.sort(followers_tensor).values
        index_percentile_90 = int(0.90 * len(sorted_followers))
        percentile_90_followers = sorted_followers[index_percentile_90].item()
    else:
        avg_followers = sum(followers_count) / len(followers_count)
        percentile_90_followers = sorted(followers_count)[int(0.90 * len(followers_count))]
    print(f"\nPromedio de followers: {avg_followers}")
    print(f"Percentil 90 de followers: {percentile_90_followers}")
else:
    print("No hay datos de followers para analizar.")

# Seleccionar los 100 actores más influyentes a partir de los datos acumulados
usuarios_lista = [{
    'user_id': uid,
    'screen_name': datos['screen_name'],
    'followers_count': datos['followers_count'],
    'total_publicaciones': datos['total_publicaciones'],
    'interacciones_totales': datos['interacciones_totales'],
    'respuestas_recibidas': datos['respuestas_recibidas'],
    'sentimientos': datos['sentimientos']
} for uid, datos in usuarios_global.items()]

usuarios_ordenados = sorted(
    usuarios_lista,
    key=lambda x: (x['followers_count'], x['respuestas_recibidas']),
    reverse=True
)
top_100_actores = usuarios_ordenados[:100]
print("\nTop 100 actores más influyentes:")
for idx, actor in enumerate(top_100_actores, start=1):
    print(f"{idx}. {actor['screen_name']} (ID: {actor['user_id']})")
    print(f"   Seguidores: {actor['followers_count']}")
    print(f"   Publicaciones sobre drogas: {actor['total_publicaciones']}")
    print(f"   Interacciones Totales: {actor['interacciones_totales']}")
    print(f"   Respuestas Recibidas: {actor['respuestas_recibidas']}")
    sentimiento_predominante = max(actor['sentimientos'], key=actor['sentimientos'].get)
    print(f"   Sentimiento predominante: {sentimiento_predominante}\n")

# Crear un grafo de interacciones entre los 10 actores principales
top_10_ids = {actor['user_id'] for actor in top_100_actores[:10]}
grafo = nx.DiGraph()
for registro in df_drogas_total.to_dict(orient='records'):
    user_id = registro['user']['id_str']
    reply_id = registro.get('in_reply_to_user_id_str')
    if reply_id and user_id in top_10_ids:
        grafo.add_node(user_id)
        grafo.add_node(reply_id)
        grafo.add_edge(user_id, reply_id)

# Asignar colores basados en el sentimiento predominante usando una paleta de Seaborn
paleta_azul = sns.color_palette("Blues", n_colors=4)
colores = []
for node in grafo.nodes():
    if node in usuarios_global:
        sentimientos = usuarios_global[node]['sentimientos']
        sentimiento_predominante = max(sentimientos, key=sentimientos.get)
        if sentimiento_predominante == 'negativa':
            colores.append(paleta_azul[0])
        elif sentimiento_predominante == 'positiva':
            colores.append(paleta_azul[3])
        elif sentimiento_predominante == 'neutral':
            colores.append(paleta_azul[2])
    else:
        colores.append(paleta_azul[1])

# Graficar el grafo
plt.figure(figsize=(12, 8))
if dispositivo.type == "cuda":
    print("Calculando posiciones de nodos en GPU...")
    try:
        positions = nx.spring_layout(grafo, seed=42)
        positions_tensor = torch.tensor(list(positions.values()), device=dispositivo)
        pos = {node: tuple(positions_tensor[i].tolist()) for i, node in enumerate(grafo.nodes())}
    except Exception as e:
        print(f"Error al calcular posiciones en GPU: {e}. Usando CPU.")
        pos = nx.spring_layout(grafo, seed=42)
else:
    pos = nx.spring_layout(grafo, seed=42)

nx.draw_networkx(grafo, pos, with_labels=False, node_color=colores, node_size=700, edge_color="gray", arrowsize=10)
plt.title("Interacciones entre los Principales Actores que hablan de drogas", fontsize=14)
output_image_path = "grafo_interacciones.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"Grafo guardado en: {output_image_path}")
plt.show()