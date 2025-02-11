# Análisis Temático de Twitter sobre Drogas en México

## Descripción del Proyecto

Este proyecto tiene como objetivo analizar grandes volúmenes de datos provenientes de la red social X (anteriormente Twitter) con el propósito de identificar publicaciones relacionadas con el tema de las drogas, clasificar su sentimiento y generar un conjunto de datos que permita mapear la información obtenida. Para ello, se implementa un flujo de procesamiento en **batches**, lo que facilita el manejo eficiente de los datos en equipos de cómputo convencionales. 

Adicionalmente, el análisis emplea **técnicas de aprendizaje profundo y procesamiento de lenguaje natural (NLP)**, incluyendo el ajuste fino (**fine-tuning**) de modelos de clasificación, con el fin de mejorar la detección de temas y el análisis de sentimiento de las publicaciones recopiladas.

---

## Metodología

El flujo de trabajo del proyecto sigue una secuencia estructurada de etapas, cada una representada por scripts específicos:

### 1. Preprocesamiento y Extracción de Datos
- `1_1_LIMPIEZA_DE_DATOS.py`: Realiza el preprocesamiento de los datos, eliminando ruido, caracteres especiales y elementos irrelevantes.
- `1_2_EXTRAER_BASE_ENTRENAMIENTO.py`: Extrae un subconjunto de datos que servirá como base de entrenamiento para los modelos de clasificación.

### 2. Detección de Palabras Clave y Modelado
- `2_1_DETECTAR_PALABRAS.py`: Identifica términos relevantes dentro de los tweets para mejorar la clasificación temática.
- `2_2_ENTRENAR_EVALUAR_MODELO_TEMA.ipynb`: Implementa y evalúa modelos de clasificación supervisada para la detección de tweets sobre drogas.
- `2_3_PREPARAR_BASE_SENTIMIENTOS.py`: Prepara el conjunto de datos necesario para entrenar el modelo de análisis de sentimiento.

### 3. Clasificación de Sentimiento
- `3_1_ENTRENAR_EVALUAR_MODELO_SENTIMIENTOS.ipynb`: Ajusta y evalúa modelos de clasificación de sentimientos con el fin de determinar la polaridad (positiva, negativa o neutral) de las publicaciones analizadas.

### 4. Aplicación de Modelos
- `4_1_APLICAR_MODELOS.py`: Implementa los modelos entrenados sobre un conjunto de datos no etiquetado para obtener predicciones de clasificación temática y de sentimiento.

### 5. Generación del Conjunto de Datos
- `5_1_CREAR_DATASET.py`: Consolida los datos procesados y genera un conjunto estructurado de información para análisis posteriores.

### 6. Visualización Geoespacial
- `6_1_CREAR_MAPA.ipynb`: Representa los resultados en un mapa para identificar patrones espaciales en la conversación sobre drogas en X.

---

## Requisitos Técnicos

Para la ejecución del proyecto, se requiere un entorno con las siguientes dependencias:

- Python 3.x
- Bibliotecas de procesamiento de datos: `pandas`, `numpy`
- Bibliotecas de procesamiento de texto: `nltk`, `spacy`, `transformers`
- Bibliotecas para modelado: `scikit-learn`, `tensorflow` o `pytorch`
- Bibliotecas para visualización de datos: `matplotlib`, `seaborn`
- Biblioteca para manipulación de archivos JSON y XML: `orjson`

Se recomienda el uso de un entorno virtual para la instalación de dependencias.  

---

## Consideraciones Finales

El proyecto tiene implicaciones tanto metodológicas como analíticas en el estudio de la conversación digital sobre drogas en México. La implementación de modelos de NLP permite identificar patrones discursivos y tendencias en el tiempo, lo que puede ser de utilidad para investigadores y tomadores de decisiones en materia de política pública y análisis de redes sociales.

Cualquier actualización del código y los modelos se documentará en este repositorio.  
