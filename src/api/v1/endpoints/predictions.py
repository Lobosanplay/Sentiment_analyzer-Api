from fastapi import APIRouter, HTTPException
from schemas.predictions_schemas import ReviewRequest, PredictionResponse
from services.model_service import model_service

router = APIRouter(tags=["predictions"])

@router.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """Predice el sentimiento de un texto en español"""
    if not model_service.model_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El servicio no está listo."
        )
    
    result, error = model_service.predict_single(request.text)
    
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    return PredictionResponse(
        sentiment=result["sentiment"],
        probability_positive=result["probability_positive"],
        probability_negative=result["probability_negative"]
    )
