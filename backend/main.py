"""
main.py
=======
Aplicación FastAPI: ciclo de vida (carga del modelo CLIP y del índice),
CORS, imágenes estáticas locales, endpoints de estado y montaje de rutas.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, embedder, indexer, utils
from .routes import image_search, text_search
from .schemas import HealthResponse, StatsResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("image_search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga los recursos pesados una sola vez al arrancar.

    1. Crea los directorios de datos.
    2. Carga el modelo CLIP en memoria (obligatorio).
    3. Inicializa el backend vectorial. Si falla, el servidor arranca igual en
       modo degradado y los endpoints de búsqueda responden HTTP 503.
    """
    logger.info("Arrancando: modelo CLIP '%s' + backend '%s' ...", utils.MODEL_NAME, indexer.BACKEND)
    utils.ensure_dirs()

    await run_in_threadpool(embedder.load_model, utils.MODEL_NAME, utils.PRETRAINED)

    try:
        await run_in_threadpool(indexer.load_index)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Índice no disponible; las búsquedas responderán 503 hasta ejecutar "
            "'python scripts/build_index.py' y reiniciar. Detalle: %s",
            exc,
        )

    yield

    if indexer.BACKEND == "pgvector":
        from .database import close_pool

        close_pool()
    logger.info("Servidor detenido.")


tags_metadata = [
    {"name": "search", "description": "Búsqueda multimodal: texto → imagen e imagen → imagen."},
    {"name": "system", "description": "Estado del servicio y estadísticas del índice."},
]

app = FastAPI(
    title="Buscador Multimodal de Imágenes",
    description=(
        "Búsqueda texto → imagen e imagen → imagen con CLIP multilingüe "
        "y PostgreSQL (pgvector) o FAISS."
    ),
    version=__version__,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Con "*" no se pueden enviar credenciales (restricción del estándar CORS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=utils.CORS_ORIGINS,
    allow_credentials="*" not in utils.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Imágenes locales del dataset (solo se usan si no hay URLs públicas).
app.mount(
    utils.IMAGE_URL_PREFIX,
    StaticFiles(directory=utils.IMAGES_DIR, check_dir=False),
    name="images",
)

app.include_router(text_search.router)
app.include_router(image_search.router)


@app.get("/", tags=["system"], include_in_schema=False)
async def root():
    return {
        "name": app.title,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Estado del servicio")
async def health() -> HealthResponse:
    """Indica si el modelo CLIP y el índice vectorial están listos."""
    size = await run_in_threadpool(indexer.index_size)
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=indexer.BACKEND,
        model=f"{utils.MODEL_NAME}/{utils.PRETRAINED}",
        model_loaded=embedder.is_ready(),
        index_loaded=indexer.is_ready(),
        index_size=size,
    )


@app.get("/stats", response_model=StatsResponse, tags=["system"], summary="Estadísticas del índice")
async def stats() -> StatsResponse:
    """Total de imágenes indexadas y conteo por categoría MS COCO."""
    data = await run_in_threadpool(indexer.stats)
    return StatsResponse(**data)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError):
    """Convierte errores de infraestructura (índice no listo, BD caída) en HTTP 503."""
    logger.error("Runtime error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})
