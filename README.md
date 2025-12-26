# Sentiment_analyzer-Api
## 📋 Descripción del Proyecto
Este proyecto es una API REST desarrollada con FastAPI que proporciona análisis de sentimientos (positivo/negativo/neutral) para textos en español. El sistema utiliza un modelo de Machine Learning basado en Regresión Logística con vectorización TF-IDF para clasificar textos y está diseñado para ser escalable, fácil de usar y adecuado tanto para análisis individuales como por lotes.

## 🎯 Características Principales
- ✅ Análisis en tiempo real de textos individuales
- 📁 Procesamiento de archivos (Excel, CSV, TXT)
- 🔄 Análisis por lotes para múltiples textos
- 📊 Generación de reportes en Excel con estadísticas
- 🏥 Endpoint de salud para monitoreo del servicio
- 📝 Validación de datos con Pydantic
- ⚡ Alta performance con FastAPI
- 🐳 Fácil despliegue y configuración

## 🏗️ Arquitectura del Proyecto
```text
📦 proyecto-sentimientos/
├── 📁 src/                     # Código fuente
│   ├── 📁 api/
│   │   └── 📁 v1/
│   │       ├── 📁 endpoints/   # Endpoints de la API
│   │       └── router.py       # Enrutador principal
│   ├── 📁 models/             # Guardar el modelo
│   ├── 📁 schemas/            # Esquemas Pydantic
│   ├── 📁 services/           # Lógica de negocio
│   └── main.py                # Punto de entrada
└── README.md                  # Documentación
```

## 🔧 Requisitos Previos
- Python 3.8+
- pip o pipenv
- Git

## 🚀 Instalación y Configuración
1. Clonar el repositorio
```bash
git clone [url-del-repositorio]
cd Sentiment_analyzer-Api
```

## 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```
## 3. Instalar dependencias
```bash
pip install fastapi uvicorn pandas scikit-learn openpyxl python-multipart huggingface_hub fsspec
# o
pip install requirements.txt
```
## 4. Ejecutar la aplicación
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 Entrenamiento del Modelo
El modelo se entrena automáticamente al iniciar la aplicación con los datos en [multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset).


## 🌐 API Endpoints
### 1. 🏥 Health Check
**GET** `/api/v1/health/status`

Verifica el estado del servicio y del modelo.

Respuesta:

```json
{
  "status": "healthy",
  "model_trained": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. 🔍 Análisis Individual
**POST** `/api/v1/predictions/predict`

Analiza el sentimiento de un texto individual.

Request:

```json
{
  "text": "El producto es excelente y de muy buena calidad"
}
```

Response:

```json
{
  "sentiment": "positivo",
  "probability_neutral": 0.16,
  "probability_positive": 0.92,
  "probability_negative": 0.08
}
```

### 3. 📦 Análisis por Lotes
**POST** `/api/v1/batch/batch-predict`

Analiza múltiples textos en una sola petición.

Request:

```json
[
  "Me encantó el servicio",
  "No volvería a comprar",
  "Calidad aceptable"
]
```

Response:

```json
{
  "predictions": [
    {
      "text": "Me encantó el servicio",
      "sentiment": "positivo",
      "probability_positive": 0.92,
      "probability_negative": 0.03,
      "probability_neutral": 0.05
    },
    {
      "text": "No volvería a comprar",
      "sentiment": "negativo",
      "probability_positive": 0.10,
      "probability_negative": 0.85,
      "probability_neutral": 0.05
    },
    {
      "text": "El producto llegó en la fecha acordada",
      "sentiment": "neutral",
      "probability_positive": 0.25,
      "probability_negative": 0.20,
      "probability_neutral": 0.55
    }
  ]
}
```

### 4. 📁 Análisis desde Archivos
**POST** `/api/v1/file/file-predictions`

Sube un archivo para análisis masivo. Soporta:

- Excel (.xlsx, .xls)
- CSV (.csv)
- Texto plano (.txt)

Parámetros:

- file: Archivo a analizar (obligatorio)
- text_column: Nombre de la columna con texto (opcional)

Response:
Devuelve un archivo Excel con:

- 📄 Hoja "Resultados": Análisis individual de cada texto
- 📊 Hoja "Resumen": Estadísticas generales

Ejemplo de resumen:

```excel
| total_reviews | positivos | negativos | porcentaje_positivos | porcentaje_negativos | porcentaje_neutrales |
|---------------|-----------|-----------|---------------------|----------------------|----------------------|
| 150           | 112       | 38        | 74.67%              | 25.33%               | 25.33%               |
```

## 🛠️ Configuración del Modelo
### Parámetros del Modelo
- Vectorizador: TF-IDF con 30000 características máximas
- Algoritmo: Regresión Logística
- Iteraciones: 4000 máximas
- Precisión típica: 80-85% (dependiendo del dataset)

## 🔍 Detección Automática de Columnas
Al subir archivos, el sistema detecta automáticamente columnas con nombres como:

- text, review, comentario, opinion
- mensaje, content, message, feedback
- review_es, comentarios

Si no encuentra coincidencias, usa la primera columna de texto disponible.

## 📊 Estadísticas Generadas
Para análisis de archivos, se incluyen:

### Métricas Principales
1. Total de reseñas: Número total de textos analizados
2. Reseñas positivas: Conteo y porcentaje
3. Reseñas negativas: Conteo y porcentaje
4. Reseñas neutrales: Conteo y porcentaje
5. Longitud promedio: Caracteres por texto
6. Confianza promedio: Certeza de las predicciones

## 🧪 Testing
### Endpoints a probar
- Health Check: Verifica que el servicio esté activo
- Predicción simple: Texto corto en ingles
- Batch processing: Array de 5-10 textos
- Archivos: Subir Excel con 100+ registros
- Errores: Textos vacíos, archivos corruptos

## 📈 Rendimiento
- Tiempo de respuesta: < 100ms para textos individuales
- Procesamiento batch: ~1000 textos/segundo
- Archivos Excel: ~10,000 filas en < 30 segundos
- Uso de memoria: Optimizado para grandes volúmenes
