import joblib
import pandas as pd
import numpy as np
from io import BytesIO
import os
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
import traceback

class ModelService:
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.model = None
            self.vectorizer = None
            self.initialized = True
    
    async def initialize(
        self, 
        model_path: str = 'models/sentiment_analyzer_model.pkl',
        create_if_missing: bool = True
    ):
        """
        Inicializa el servicio cargando el modelo.
        """
        try:
            from services.model_creator_service import create_model
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if not os.path.exists(model_path) and create_if_missing:
                print("Modelo no encontrado, creando nuevo...")
                success, message = await create_model(model_path)
                if not success:
                    raise Exception(f"Error creando modelo: {message}")
            

            self.model = joblib.load(model_path)
            
            vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
            
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                print(f"Vectorizador cargado: {vectorizer_path}")
            else:
                print(f"⚠️ Vectorizador no encontrado en: {vectorizer_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
            traceback.print_exc()
            return False

    def predict_single(self, text: str):
        """Predice el sentimiento de un solo texto"""
        if not self.model or not self.vectorizer:
            return None, "Modelo o vectorizador no cargado"
        
        try:
            if not text or not isinstance(text, str) or text.strip() == "":
                return {
                    "sentiment": "neutral",
                    "probability_neutral": 1.0,
                    "probability_positive": 0.0,
                    "probability_negative": 0.0,
                    "text": text
                }, None
            
            text_clean = text.strip()
            text_vectorized = self.vectorizer.transform([text_clean])
            prediction = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            
            if prediction == 0:
                sentiment = "negative"
            elif prediction == 1:
                sentiment = "neutral"
            elif prediction == 2:
                sentiment = "positive"
            else:
                sentiment = "neutral"
            
            return {
                "text": text_clean,
                "sentiment": sentiment,
                "probability_negative": float(probabilities[0]),
                "probability_neutral": float(probabilities[1]),
                "probability_positive": float(probabilities[2])
            }, None
            
        except Exception as e:
            return None, f"Error en predicción: {str(e)}"
    
    def predict_batch(self, texts: list[str]):
        """Predice sentimientos para múltiples textos"""
        if not self.model or not self.vectorizer:
            return None, "Modelo o vectorizador no cargado"
        
        try:
            texts_clean = []
            for text in texts:
                if pd.notna(text) and isinstance(text, str):
                    cleaned = text.strip()
                    if cleaned:
                        texts_clean.append(cleaned)
            
            if not texts_clean:
                return [], "No hay textos válidos para analizar"
            
            texts_vectorized = self.vectorizer.transform(texts_clean)
            predictions = self.model.predict(texts_vectorized)
            probabilities = self.model.predict_proba(texts_vectorized)
            
            results = []
            for i, text in enumerate(texts_clean):
                if predictions[i] == 0:
                    sentiment = "negative"
                elif predictions[i] == 1:
                    sentiment = "neutral"
                elif predictions[i] == 2:
                    sentiment = "positive"
                else:
                    sentiment = "neutral"
                
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "probability_negative": float(probabilities[i][0]),
                    "probability_neutral": float(probabilities[i][1]),
                    "probability_positive": float(probabilities[i][2])
                })
            
            return results, None
            
        except Exception as e:
            print(f"Error en predict_batch: {str(e)}")
            traceback.print_exc()
            return None, f"Error en predicción batch: {str(e)}"

    def predict_from_file(self, file_content: bytes, filename: str, text_column: str = None):
        if not self.model or not self.vectorizer:
            return None, "Modelo o vectorizador no cargado"
        
        try:
            file_extension = filename.split('.')[-1].lower() if '.' in filename else 'txt'
            
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(BytesIO(file_content))
            elif file_extension == 'csv':
                df = pd.read_csv(BytesIO(file_content))
            else:
                content = file_content.decode('utf-8', errors='ignore')
                lines = content.split('\n')
                df = pd.DataFrame(lines, columns=['text'])
            
            if text_column:
                if text_column not in df.columns:
                    return None, f"La columna '{text_column}' no existe en el archivo"
                selected_column = text_column
            else:
                possible_names = ['text', 'review', 'comentario', 'opinion', 'mensaje', 
                                'content', 'message', 'feedback', 'review_es', 'comentarios']
                
                selected_column = None
                for col in df.columns:
                    col_lower = col.lower()
                    for name in possible_names:
                        if name in col_lower:
                            selected_column = col
                            print(f"Columna encontrada automáticamente: {selected_column}")
                            break
                    if selected_column:
                        break
                
                if not selected_column:
                    for col in df.columns:
                        if pd.api.types.is_string_dtype(df[col]):
                            selected_column = col
                            print(f"Usando primera columna de texto: {selected_column}")
                            break
                
                if not selected_column and len(df.columns) > 0:
                    selected_column = df.columns[0]
                    print(f"Usando primera columna: {selected_column}")
            
            df = df.dropna(subset=[selected_column])
            df[selected_column] = df[selected_column].astype(str).str.strip()
            df = df[df[selected_column] != '']
            
            if df.empty:
                return {
                    "results": [],
                    "summary": {
                        "total_reviews": 0,
                        "neutrals": 0,
                        "positives": 0,
                        "negatives": 0,
                        "neutral_percentage": 0,
                        "positive_percentage": 0,
                        "negative_percentage": 0
                    }
                }, None
            
            texts = df[selected_column].tolist()
            predictions, error = self.predict_batch(texts)
            
            if error:
                return None, error
            
            total_reviews = len(predictions)
            positive_count = sum(1 for p in predictions if p['sentiment'] == 'positive')
            negative_count = sum(1 for p in predictions if p['sentiment'] == 'negative')
            neutral_count = sum(1 for p in predictions if p['sentiment'] == 'neutral')
            
            summary = {
                "total_reviews": total_reviews,
                "neutrals": neutral_count,
                "positives": positive_count,
                "negatives": negative_count,
                "neutral_percentage": round((neutral_count / total_reviews) * 100, 2) if total_reviews > 0 else 0,
                "positive_percentage": round((positive_count / total_reviews) * 100, 2) if total_reviews > 0 else 0,
                "negative_percentage": round((negative_count / total_reviews) * 100, 2) if total_reviews > 0 else 0
            }
            
            return {
                "results": predictions,
                "summary": summary,
            }, None
            
        except Exception as e:
            print(f"Error en predict_from_file: {str(e)}")
            traceback.print_exc()
            return None, f"Error al procesar archivo: {str(e)}"
    
    def create_excel(self, result):
        try:
            df = pd.DataFrame(result["results"])
            
            df['prediction_index'] = range(1, len(df) + 1)
            df['text_length'] = df['text'].str.len()
            df['confidence'] = df[['probability_positive', 'probability_negative', 'probability_neutral']].max(axis=1)
            
            summary_df = pd.DataFrame([result["summary"]])
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Resultados')
                summary_df.to_excel(writer, index=False, sheet_name='Resumen')
            
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": "attachment; filename=resultados_analisis.xlsx"
                }
            )
            
        except Exception as e:
            print(f"Error creando Excel: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error generando archivo Excel: {str(e)}")

model_service = ModelService()