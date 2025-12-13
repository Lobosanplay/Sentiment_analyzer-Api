from fastapi import APIRouter
from api.v1.endpoints import health, predictions, batch_predictions, file_predictions

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(batch_predictions.router, prefix="/batch", tags=["batch predictions"])
api_router.include_router(file_predictions.router, prefix='/file', tags=["file predictions"])