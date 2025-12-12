from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    probability_positive: float
    probability_negative: float
    
class BatchPredictionItem(BaseModel):
    text: str
    sentiment: str
    probability_positive: float
    probability_negative: float

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]