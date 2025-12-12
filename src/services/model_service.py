import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ModelService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_trained = False
    
    def train_model(self, excel_path: str = '../data/BBDD.xlsx'):
        """Entrena el modelo con datos del Excel"""
        try:
            df = pd.read_excel(excel_path)
            
            df = df[['sentimiento', 'review_es']].copy()
            target_map = {'positivo': 1, 'negativo': 0}
            df['target'] = df['sentimiento'].map(target_map)
            
            self.vectorizer = TfidfVectorizer(max_features=2000)
            X = self.vectorizer.fit_transform(df['review_es'].astype(str))
            
            self.model = LogisticRegression(max_iter=2000)
            self.model.fit(X, df['target'])
            
            train_accuracy = self.model.score(X, df['target'])
            print(f"Modelo entrenado. Accuracy: {train_accuracy:.2%}")
            
            self.model_trained = True
            return True, "Modelo entrenado exitosamente"
            
        except Exception as e:
            self.model_trained = False
            return False, f"Error al entrenar modelo: {str(e)}"
    
    def predict_single(self, text: str):
        """Predice el sentimiento de un solo texto"""
        if not self.model_trained:
            return None, "Modelo no entrenado"
        
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        
        sentiment = "positivo" if prediction == 1 else "negativo"
        
        return {
            "sentiment": sentiment,
            "probability_positive": float(probabilities[1]),
            "probability_negative": float(probabilities[0])
        }, None
    
    def predict_batch(self, texts: list[str]):
        """Predice sentimientos para múltiples textos"""
        if not self.model_trained:
            return None, "Modelo no entrenado"
        
        texts_vectorized = self.vectorizer.transform(texts)
        predictions = self.model.predict(texts_vectorized)
        probabilities = self.model.predict_proba(texts_vectorized)
        
        results = []
        for i, text in enumerate(texts):
            sentiment = "positivo" if predictions[i] == 1 else "negativo"
            results.append({
                "text": text,
                "sentiment": sentiment,
                "probability_positive": float(probabilities[i][1]),
                "probability_negative": float(probabilities[i][0])
            })
        
        return results, None

model_service = ModelService()
