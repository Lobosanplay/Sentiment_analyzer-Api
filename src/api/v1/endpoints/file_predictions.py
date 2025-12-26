from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from services.model_service import model_service

router = APIRouter(tags=["file predictions"])

@router.post('/file-predictions')
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
        contents = await file.read()
        
        result, error = model_service.predict_from_file(contents, file.filename, text_column)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        exel = model_service.create_excel(result)

        return exel

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

