from fastapi import APIRouter, HTTPException
from typing import List
from schemas.predictions_schemas import BatchPredictionResponse, BatchPredictionItem
from services.model_service import model_service

router = APIRouter(tags=["batch predictions"])

@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(texts: List[str]):
    """Predice sentimientos para múltiples textos"""
    if not model_service.model_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El servicio no está listo."
        )
    
    results, error = model_service.predict_batch(texts)
    
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    predictions = [
        BatchPredictionItem(
            text=item["text"],
            sentiment=item["sentiment"],
            probability_neutral=item["probability_neutral"],
            probability_positive=item["probability_positive"],
            probability_negative=item["probability_negative"]
        )
        for item in results
    ]
    
    return BatchPredictionResponse(predictions=predictions)
