"""``POST /search/image``: búsqueda imagen → imagen."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import embedder, indexer, utils
from ..schemas import SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

# 10 MB es más que suficiente para una imagen de consulta.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post(
    "/search/image",
    response_model=SearchResponse,
    summary="Buscar imágenes por imagen (imagen → imagen)",
    description=(
        "Sube una imagen de consulta; se codifica con el encoder visual de CLIP "
        "y se devuelven las imágenes visualmente más similares."
    ),
    responses={
        400: {"description": "Archivo vacío, demasiado grande, tipo no soportado o corrupto."},
        503: {"description": "El índice vectorial no está listo."},
    },
)
async def search_image(
    file: UploadFile = File(..., description="Imagen de consulta (JPG/PNG/WebP/BMP). Máx. 10 MB."),
    top_k: int = Form(
        default=utils.TOP_K_DEFAULT,
        ge=1,
        le=utils.TOP_K_MAX,
        description="Número de resultados a devolver.",
    ),
) -> SearchResponse:
    if not indexer.is_ready():
        raise HTTPException(
            status_code=503,
            detail="El índice vectorial no está listo. Construye el índice y reinicia el servidor.",
        )

    if file.content_type and file.content_type.lower() not in utils.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado '{file.content_type}'. Sube una imagen JPG, PNG o WebP.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="La imagen es demasiado grande (máximo 10 MB).")

    try:
        image = utils.load_image_from_bytes(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()

    embedding = await run_in_threadpool(embedder.get_image_embedding, image)
    hits = await run_in_threadpool(indexer.search, embedding, top_k)
    results = utils.format_results(hits)

    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Búsqueda por imagen '%s' -> %d resultados en %.1f ms", file.filename, len(results), took_ms)

    return SearchResponse(results=results, count=len(results), took_ms=round(took_ms, 1))
