
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
    """
    Gestiona el ciclo de vida de la aplicación FastAPI (startup y shutdown).

    Al **inicio** (startup):
    1. Crea los directorios de datos necesarios (``utils.ensure_dirs()``).
    2. Carga el modelo CLIP en memoria (``embedder.load_model()``).
    3. Inicializa el índice de búsqueda / conexión a PostgreSQL
       (``indexer.load_index()``). Si falla, el servidor sigue arrancando
       en modo degradado y los endpoints de búsqueda retornan HTTP 503
       hasta que el índice sea construido manualmente.

    Al **cierre** (shutdown): registra el evento en el logger.

    Parameters
    ----------
    app : fastapi.FastAPI
        Instancia de la aplicación FastAPI inyectada automáticamente por
        el framework al registrar el lifespan.

    Yields
    ------
    None
        Punto de suspensión entre startup y shutdown; la aplicación
        atiende requests mientras está suspendida aquí.

    Notes
    -----
    - Debe registrarse como ``lifespan=lifespan`` en el constructor de
      ``FastAPI()``, no como evento ``@app.on_event`` (patrón moderno).
    - Los errores de ``indexer.load_index()`` se capturan y logean como
      advertencia para permitir que el servidor inicie de todas formas.
    """
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
    """
    Endpoint de verificación de estado (health check) del servidor.

    Retorna el estado operativo de los componentes críticos:
    modelo CLIP e indexador de búsqueda. Útil para monitoreo,
    load balancers y scripts de despliegue.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    dict
        Diccionario JSON con las siguientes claves:

        - ``"status"`` (str): siempre ``"ok"`` si el servidor está corriendo.
        - ``"model_loaded"`` (bool): ``True`` si el modelo CLIP está en memoria.
        - ``"index_loaded"`` (bool): ``True`` si el indexador/BD está listo.
        - ``"index_size"`` (int): número de imágenes indexadas actualmente.
    """
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
    """
    Manejador global de excepciones ``RuntimeError`` para toda la aplicación.

    Convierte errores de tiempo de ejecución no capturados (principalmente
    el lanzado por ``indexer.search()`` cuando la BD no está lista) en
    respuestas HTTP 503 con un mensaje descriptivo en el cuerpo JSON.

    Parameters
    ----------
    _request : fastapi.Request
        Objeto de la petición HTTP que desencadenó el error. No se utiliza
        directamente, pero es requerido por la firma del handler de FastAPI.
    exc : RuntimeError
        Instancia de la excepción capturada. Su mensaje (``str(exc)``) se
        incluye en el campo ``"detail"`` de la respuesta JSON.

    Returns
    -------
    fastapi.responses.JSONResponse
        Respuesta HTTP con:
        - Código de estado: ``503 Service Unavailable``.
        - Cuerpo: ``{"detail": "<mensaje del error>"}``.
    """
    logger.error("Runtime error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})
