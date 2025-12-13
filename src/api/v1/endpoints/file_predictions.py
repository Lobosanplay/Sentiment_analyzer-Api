from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from schemas.predictions_schemas import FilePredictionItem
from services.model_service import model_service

router = APIRouter(tags=["file predictions"])

@router.post('/file-predictions', response_model=FilePredictionItem)
async def file_prediction(
    file: UploadFile = File(...),
    text_column: Optional[str] = None
):
    """
    Analiza sentimientos desde un archivo (Excel, CSV o TXT)
    
    Args:
        file: Archivo a analizar
        text_column: Nombre de la columna que contiene el texto (opcional)
    """
    try:
        if not model_service.model_trained:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. El servicio no está listo."
            )

        contents = await file.read()
        filename = file.filename
        
        result, error = model_service.predict_from_file(contents, filename, text_column)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

