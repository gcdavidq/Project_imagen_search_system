"""Esquemas Pydantic compartidos por los endpoints de la API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import utils


class TextSearchRequest(BaseModel):
    """Cuerpo de la petición para ``POST /search/text``."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Texto descriptivo para buscar imágenes (cualquier idioma).",
        examples=["un perro jugando con una pelota"],
    )
    top_k: int = Field(
        default=utils.TOP_K_DEFAULT,
        ge=1,
        le=utils.TOP_K_MAX,
        description="Número de resultados a devolver.",
        examples=[6],
    )


class SearchResult(BaseModel):
    """Un único resultado de búsqueda."""

    image_url: str = Field(description="URL de la imagen (pública o servida por esta API).")
    score: float = Field(description="Similitud coseno entre la consulta y la imagen. Más alto es mejor.")
    categories: list[str] = Field(default_factory=list, description="Categorías MS COCO de la imagen.")


class SearchResponse(BaseModel):
    """Respuesta compartida por los endpoints de búsqueda por texto e imagen."""

    results: list[SearchResult] = Field(description="Resultados ordenados de mayor a menor similitud.")
    count: int = Field(description="Cantidad de resultados retornados.")
    took_ms: float = Field(description="Tiempo de embedding + búsqueda vectorial, en milisegundos.")


class CategoryCount(BaseModel):
    name: str
    count: int


class StatsResponse(BaseModel):
    """Respuesta de ``GET /stats``."""

    backend: str = Field(description="Backend vectorial activo: 'pgvector' o 'faiss'.")
    total_images: int
    categories: list[CategoryCount] = Field(description="Conteo de imágenes por categoría, descendente.")


class HealthResponse(BaseModel):
    """Respuesta de ``GET /health``."""

    status: str
    version: str
    backend: str
    model: str
    model_loaded: bool
    index_loaded: bool
    index_size: int
