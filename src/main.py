from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="API para clasificar sentimientos positivos/negativos en textos en español",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    probability_positive: float
    probability_negative: float
    
model = None
vectorizer = None
model_trained = False

@app.on_event("startup")
async def startup_event():
    """Inicializa el modelo al iniciar la API"""
    global model, vectorizer, model_trained
    
    try: 
        df = pd.read_excel('../data/BBDD.xlsx')
        
        df = df[['sentimiento', 'review_es']].copy()
        target_map = {'positivo': 1, 'negativo': 0}
        df['target'] = df['sentimiento'].map(target_map)
        
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(df['review_es'].astype(str))
        
        model = LogisticRegression(max_iter=2000)
        model.fit(X, df['target'])
        
        train_accuracy = model.score(X, df['target'])
        print(f"✅ Modelo entrenado. Accuracy en entrenamiento: {train_accuracy:.2%}")
        
        model_trained = True
        print("🎉 ¡Modelo listo para hacer predicciones!")
        
    except Exception as e:
        vectorizer = TfidfVectorizer(max_features=2000)
        model = LogisticRegression(max_iter=2000)
        print('Error')
        return f"Error al cargar/entrenar modelo: {e}"
        
@app.get('/home')
def read_root():
    return {"message": "Bienvenido a la API para analizar sentimientos"}

@app.get("/health")
def health_check():
    """Endpoint para verificar estado de la API"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post('/predict', response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        text_vetorized = vectorizer.transform([request.text])
        
        prediction = model.predict(text_vetorized)[0]
        probabilities = model.predict_proba(text_vetorized)[0]
        
        sentiment = "positivo" if prediction == 1 else "negativo"

        return PredictionResponse(
            sentiment=sentiment,
            probability_positive=float(probabilities[1]),
            probability_negative=float(probabilities[0])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")