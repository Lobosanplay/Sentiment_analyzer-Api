from fastapi import APIRouter
from services.model_service import model_service

router = APIRouter(tags=["health"])

@router.get("/health")
def health_check():
    """Endpoint para verificar estado de la API"""
    return {
        "status": "healthy",
        "model_loaded": model_service.model_trained,
        "api_version": "v1"
    }

@router.get("/model-status")
def model_status():
    """Información del estado del modelo"""
    return {
        "model_trained": model_service.model_trained,
        "model_ready": model_service.model is not None,
        "vectorizer_ready": model_service.vectorizer is not None
    }
