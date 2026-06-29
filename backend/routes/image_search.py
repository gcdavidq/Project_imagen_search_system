
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import embedder, indexer, utils
from .text_search import SearchResponse  # reutilizar el esquema de respuesta compartido

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

# Rechazar subidas absurdamente grandes temprano (10 MB es suficiente para una imagen de consulta).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post(
    "/search/image", 
    response_model=SearchResponse,
    summary="Buscar imágenes por imagen (Image-to-Image)",
    description="Sube una imagen como consulta para encontrar las imágenes más similares semánticamente y visualmente en la base de datos PostgreSQL usando embeddings de CLIP.",
    responses={
        400: {"description": "Error en el archivo (ej. formato no soportado, vacío, muy grande, bytes corruptos)."},
        503: {"description": "El índice de la base de datos no está listo o hay problemas de conexión."}
    }
)
async def search_image(
    file: UploadFile = File(..., description="Imagen de consulta (JPG/PNG/WebP). Máx 10MB."),
    top_k: int = Form(default=utils.TOP_K_DEFAULT, ge=1, le=50, description="Número de resultados a devolver (1-50)."),
) -> SearchResponse:
    """Retorna las imágenes de la base de datos más similares a una imagen subida."""
    if not indexer.is_ready():
        raise HTTPException(
            status_code=503,
            detail="La conexión a la base de datos PostgreSQL no está lista. Asegúrate de inicializarla primero.",
        )

    # Validar el tipo de contenido declarado (el mejor esfuerzo -- la decodificación real ocurre abajo).
    if file.content_type and file.content_type.lower() not in utils.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado '{file.content_type}'. Sube una imagen JPG, PNG o WebP.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="La imagen subida es demasiado grande (máximo 10 MB).")

    # Decodificar y normalizar la imagen; bytes inválidos -> HTTP 400.
    try:
        image = utils.load_image_from_bytes(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Limitar top_k al rango soportado.
    safe_top_k = max(1, min(int(top_k), 50))

    started = time.perf_counter()

    embedding = embedder.get_image_embedding(image)
    # Ejecutamos la búsqueda de base de datos síncrona en un hilo separado para no bloquear el Event Loop de FastAPI
    hits = await run_in_threadpool(indexer.search, embedding, top_k=safe_top_k)
    results = utils.format_results(hits)

    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Búsqueda por imagen '%s' -> %d resultados en %.1f ms",
                file.filename, len(results), took_ms)

    return SearchResponse(results=results, count=len(results), took_ms=round(took_ms, 1))
