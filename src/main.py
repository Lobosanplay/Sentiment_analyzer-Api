from fastapi import FastAPI
from api.v1.router import api_router
from services.model_service import model_service

app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="API para clasificar sentimientos positivos/negativos en textos en español",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Inicializa el modelo al iniciar la API"""
    await model_service.initialize()

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    """Página principal de la API"""
    return {
        "message": "Bienvenido a la API de Análisis de Sentimientos",
        "documentation": "/docs",
        "api_version": "v1",
        "endpoints": {
            "home": "/",
            "health": "/api/v1/health/health",
            "model_status": "/api/v1/health/model-status",
            "predict": "/api/v1/predictions/predict",
            "batch_predict": "/api/v1/batch/batch-predict",
            "filea_prediction": "/api/v1/file/fil-prediction"
        }
    }