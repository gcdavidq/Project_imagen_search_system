"""``POST /search/text``: búsqueda texto → imagen."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from .. import embedder, indexer, utils
from ..schemas import SearchResponse, TextSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post(
    "/search/text",
    response_model=SearchResponse,
    summary="Buscar imágenes por texto (texto → imagen)",
    description=(
        "Convierte el texto en un embedding CLIP y devuelve las imágenes más "
        "similares por coseno. Acepta consultas en español, inglés y otros idiomas."
    ),
    responses={
        400: {"description": "Consulta vacía."},
        503: {"description": "El índice vectorial no está listo."},
    },
)
async def search_text(request: TextSearchRequest) -> SearchResponse:
    if not indexer.is_ready():
        raise HTTPException(
            status_code=503,
            detail="El índice vectorial no está listo. Construye el índice y reinicia el servidor.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La consulta no debe estar vacía.")

    started = time.perf_counter()

    embedding = await run_in_threadpool(embedder.get_text_embedding, query)
    hits = await run_in_threadpool(indexer.search, embedding, request.top_k)
    results = utils.format_results(hits)

    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Búsqueda de texto '%s' -> %d resultados en %.1f ms", query, len(results), took_ms)

    return SearchResponse(results=results, count=len(results), took_ms=round(took_ms, 1))
