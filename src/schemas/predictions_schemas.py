from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    probability_neutral: float
    probability_positive: float
    probability_negative: float
    
class BatchPredictionItem(BaseModel):
    text: str
    sentiment: str
    probability_neutral: float
    probability_positive: float
    probability_negative: float

class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]
    
class FilePredictionItem(BaseModel):
    text: str
    sentiment: str
    probability_positive: float
    probability_negative: float

class FilePredictionResponse(BaseModel):
    predictions: List[FilePredictionItem]

class AnalysisSummary(BaseModel):
    total_reviews: int
    positivos: int
    negativos: int
    porcentaje_positivos: float
    porcentaje_negativos: float