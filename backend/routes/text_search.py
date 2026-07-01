from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional, List

from .. import embedder, indexer, utils

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


class TextSearchRequest(BaseModel):
    """Cuerpo de la petición para ``POST /search/text``."""

    query: str = Field(..., min_length=1, description="Texto descriptivo para buscar imágenes.", examples=["Un perro jugando con una pelota"])
    top_k: int = Field(
        default=utils.TOP_K_DEFAULT,
        ge=1,
        le=50,
        description="Número de resultados a devolver.",
        examples=[5]
    )


class SearchResult(BaseModel):
    """Un único resultado de búsqueda."""

    image_url: str = Field(description="URL estática para cargar la imagen.")
    score: float = Field(description="Puntuación de similitud semántica (distancia del coseno invertida). Más alto es mejor.")
    categories: Optional[List[str]] = Field(default=None, description="Categorías detectadas en la imagen, obtenidas de Supabase.")


class SearchResponse(BaseModel):
    """Estructura de respuesta compartida por los endpoints de búsqueda de texto e imagen."""

    results: List[SearchResult] = Field(description="Lista ordenada de resultados más similares.")
    count: int = Field(description="Cantidad de resultados retornados.")
    took_ms: float = Field(description="Tiempo que tomó realizar la búsqueda vectorial y el embedding (en milisegundos).")


@router.post(
    "/search/text", 
    response_model=SearchResponse,
    summary="Buscar imágenes por texto (Text-to-Image)",
    description="Convierte un texto de consulta en un embedding usando CLIP y busca las imágenes más similares en la base de datos PostgreSQL.",
    responses={
        400: {"description": "Consulta vacía o inválida."},
        503: {"description": "El índice de la base de datos no está listo o hay problemas de conexión."}
    }
)
async def search_text(request: TextSearchRequest) -> SearchResponse:
    """
    Busca imágenes por similitud semántica usando una consulta de texto libre.

    Convierte el texto de entrada en un embedding vectorial mediante el modelo
    CLIP multilingüe, luego ejecuta una búsqueda por similitud del coseno en
    la base de datos PostgreSQL (pgvector) y retorna las ``top_k`` imágenes
    más similares en orden descendente de relevancia.

    Parameters
    ----------
    request : TextSearchRequest
        Cuerpo de la petición JSON con los campos:

        - ``query`` (str): texto descriptivo de lo que se desea encontrar.
          Mínimo 1 carácter, soporta múltiples idiomas.
        - ``top_k`` (int, opcional): número de resultados a retornar
          (entre 1 y 50; por defecto ``utils.TOP_K_DEFAULT``).

    Returns
    -------
    SearchResponse
        Respuesta JSON con los campos:

        - ``results`` (List[SearchResult]): lista ordenada de imágenes
          más similares, cada una con ``image_url``, ``score`` y
          ``categories`` opcionales.
        - ``count`` (int): número de resultados retornados.
        - ``took_ms`` (float): tiempo total de embedding + búsqueda
          en milisegundos.

    Raises
    ------
    HTTPException (503)
        Si el indexador/base de datos no está listo (``indexer.is_ready()``
        retorna ``False``).
    HTTPException (400)
        Si la consulta, luego de aplicar ``.strip()``, queda vacía.

    Notes
    -----
    - La búsqueda en BD es síncrona; se ejecuta en un hilo separado
      mediante ``run_in_threadpool`` para no bloquear el event loop.
    - El tiempo medido incluye tanto el embedding del texto como la
      consulta vectorial en PostgreSQL.
    """
    if not indexer.is_ready():
        raise HTTPException(
            status_code=503,
            detail="La conexión a la base de datos PostgreSQL no está lista. Asegúrate de inicializarla primero.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La consulta no debe estar vacía.")

    started = time.perf_counter()

    embedding = embedder.get_text_embedding(query)
    # Ejecutamos la búsqueda de base de datos síncrona en un hilo separado
    hits = await run_in_threadpool(indexer.search, embedding, top_k=request.top_k)
    results = utils.format_results(hits)

    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Búsqueda de texto '%s' -> %d resultados en %.1f ms", query, len(results), took_ms)

    return SearchResponse(results=results, count=len(results), took_ms=round(took_ms, 1))
