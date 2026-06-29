
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import embedder, indexer, utils
from .routes import image_search, text_search

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("image_search")


# --------------------------------------------------------------------------- #
# Lifespan: load heavy resources once
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CLIP model and PostgreSQL connection on start-up; release on shutdown."""
    logger.info("Starting up: loading CLIP model and DB connection ...")
    utils.ensure_dirs()

    # 1) CLIP model -- always required.
    embedder.load_model(utils.MODEL_NAME, utils.PRETRAINED)

    # 2) DB connection -- optional at boot. The server still starts so that the
    #    pages render and return a clean 503 until the index is built.
    try:
        indexer.load_index()
    except Exception as e:
        logger.warning(
            f"Database not fully initialized. Search endpoints will return HTTP 503 until "
            f"you run 'python scripts/build_index.py' and restart. Detail: {e}"
        )

    yield

    logger.info("Shutting down.")



tags_metadata = [
    {
        "name": "search",
        "description": "Operaciones de búsqueda multimodal (Texto e Imagen).",
    }
]

app = FastAPI(
    title="Multimodal Image Search API",
    description="Motor de búsqueda de imágenes texto-a-imagen e imagen-a-imagen alimentado por CLIP y PostgreSQL (pgvector).",
    version="2.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Soporte API",
        "url": "http://localhost:8000",
    },
    lifespan=lifespan,
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    utils.IMAGE_URL_PREFIX,
    StaticFiles(directory=utils.IMAGES_DIR, check_dir=False),
    name="images",
)

#Rutas
app.include_router(text_search.router)
app.include_router(image_search.router)


@app.get("/health")
async def health():
    """Lightweight readiness probe for monitoring / load balancers."""
    return {
        "status": "ok",
        "model_loaded": embedder.is_ready(),
        "index_loaded": indexer.is_ready(),
        "index_size": indexer.index_size(),
    }


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request: Request, exc: RuntimeError):
    """Surface unexpected runtime errors (e.g. DB not loaded) as HTTP 503."""
    logger.error("Runtime error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})
