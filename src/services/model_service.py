import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

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
            
            texts_vectorized = self.vectorizer.transform(texts_clean)
            
            predictions = self.model.predict(texts_vectorized)
            probabilities = self.model.predict_proba(texts_vectorized)
            
            results = []
            for i, text in enumerate(texts_clean):
                sentiment = "positivo" if predictions[i] == 1 else "negativo"
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "probability_positive": float(probabilities[i][1]),
                    "probability_negative": float(probabilities[i][0])
                })
            
            return results, None
            
        except Exception as e:
            import traceback
            print(f"Error detallado en predict_batch: {str(e)}")
            print(traceback.format_exc())
            return None, f"Error en predicción batch: {str(e)}"

    def predict_from_file(self, file_content: bytes, filename: str, text_column: str):
        if not self.model_trained:
            return None, "Modelo no entrenado"
        
        try:
            file_extension = filename.split('.')[-1].lower() if '.' in filename else 'txt'

            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(BytesIO(file_content))
            elif file_extension == 'csv':
                df = pd.read_csv(BytesIO(file_content))
            else:
                content = file_content.decode('utf-8')
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
                    
                    if not selected_column:
                        selected_column = df.columns[0]
            
            df = df.dropna(subset=[selected_column])
            df[selected_column] = df[selected_column].astype(str).str.strip()
            df = df[df[selected_column] != '']

            if df.empty:
                return {
                    "results": [],
                    "summary": {
                        "total_reviews": 0,
                        "positivos": 0,
                        "negativos": 0,
                        "porcentaje_positivos": 0,
                        "porcentaje_negativos": 0
                    }
                }, None

            texts = df[selected_column].tolist()
            predictions, error = self.predict_batch(texts)

            if error:
                return None, error

            total_reviews = len(predictions)
            positive_count = sum(1 for p in predictions if p['sentiment'] == 'positivo')
            negative_count = total_reviews - positive_count

            summary = {
                "total_reviews": total_reviews,
                "positivos": positive_count,
                "negativos": negative_count,
                "porcentaje_positivos": (positive_count / total_reviews) * 100 if total_reviews > 0 else 0,
                "porcentaje_negativos": (negative_count / total_reviews) * 100 if total_reviews > 0 else 0
            }
            
            return {
                "results": predictions,
                "summary": summary,
            }, None
            
        except Exception as e:
            return None, f"Error al procesar archivo: {str(e)}"
        
    def create_exel(self, result):
        try:
            df = pd.DataFrame(result["results"])
            
            df['prediction_index'] = range(len(df))
            df['text_length'] = df['text'].str.len()
            df['confidence'] = df[['probability_positive', 'probability_negative']].max(axis=1)
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Resultados')

                summary_df = pd.DataFrame([result["summary"]])
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
            raise HTTPException(status_code=500, detail=str(e))


model_service = ModelService()
