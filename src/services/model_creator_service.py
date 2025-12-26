import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

class VectorizerModel:
    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )
        self.is_fitted = False
    
    def fit_transform(self, text_series: pd.Series) -> np.ndarray:
        vectors = self.vectorizer.fit_transform(text_series)
        self.is_fitted = True
        return vectors.toarray()
    
    def save_vectorizer(self, path: str) -> None:
        if self.is_fitted:
            joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")
        else:
            raise ValueError("El vectorizador no ha sido entrenado")
    
    def transform(self, text_series: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("El vectorizador no ha sido entrenado")
        vectors = self.vectorizer.transform(text_series)
        return vectors.toarray()

async def create_model(model_path, max_features: int = 5000):
    """Entrena el modelo con datos del Excel"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        df = pd.read_csv('hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/train_df.csv')
        
        vectorizer_model = VectorizerModel(max_features=max_features)
        vectors = vectorizer_model.fit_transform(df['text'].astype(str))

        vectorizer_model.save_vectorizer(model_path.replace('.pkl', ''))
        
        model = LogisticRegression(max_iter=4000, random_state=42)
        model.fit(vectors, df['label'])
        
        joblib.dump(model, model_path)
        
        return True, "Modelo entrenado y guardado exitosamente"
        
    except Exception as e:
        import traceback
        print(f"Error detallado: {traceback.format_exc()}")
        return False, f"Error al entrenar modelo: {str(e)}"