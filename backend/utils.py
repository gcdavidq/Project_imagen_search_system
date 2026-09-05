"""
utils.py
========
Configuración centralizada (leída desde el entorno / ``.env``) y ayudantes
reutilizables compartidos por el backend: carga de imágenes, listado de
archivos y formateo de respuestas.
"""

from __future__ import annotations

import io
import logging
import os

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

# Cargar variables desde un archivo .env local si existe.
load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #

# La raíz del proyecto es el padre del paquete ``backend``.
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ``DATA_DIR`` se puede sobreescribir por entorno; todo lo demás deriva de él.
DATA_DIR: str = os.getenv("DATA_DIR") or os.path.join(BASE_DIR, "data")
IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "embeddings")
METADATA_PATH: str = os.path.join(DATA_DIR, "metadata.json")

# Archivos del modo FAISS (offline).
FAISS_INDEX_PATH: str = os.path.join(EMBEDDINGS_DIR, "index.faiss")
FAISS_PATHS_PATH: str = os.path.join(EMBEDDINGS_DIR, "image_paths.json")

# --------------------------------------------------------------------------- #
# Modelo CLIP
# --------------------------------------------------------------------------- #

# Modelo multilingüe (OpenCLIP): torre de texto XLM-RoBERTa + ViT-B/32.
# Permite consultas en español, inglés y decenas de idiomas más. 512 dims.
MODEL_NAME: str = os.getenv("MODEL_NAME", "xlm-roberta-base-ViT-B-32")
PRETRAINED: str = os.getenv("PRETRAINED", "laion5b_s13b_b90k")
EMBEDDING_DIM: int = 512

# --------------------------------------------------------------------------- #
# Búsqueda
# --------------------------------------------------------------------------- #

# "pgvector" (PostgreSQL) o "faiss" (índice local en disco).
INDEX_BACKEND: str = os.getenv("INDEX_BACKEND", "pgvector").strip().lower()
if INDEX_BACKEND not in {"pgvector", "faiss"}:
    raise ValueError(
        f"INDEX_BACKEND='{INDEX_BACKEND}' no es válido. Usa 'pgvector' o 'faiss'."
    )

TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "6"))
TOP_K_MAX: int = 50

# Prefijo de URL bajo el cual se sirven las imágenes locales (ver main.py).
IMAGE_URL_PREFIX: str = "/images"

# Tipos MIME aceptados por el endpoint de subida.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}

# --------------------------------------------------------------------------- #
# Base de datos (solo INDEX_BACKEND=pgvector)
# --------------------------------------------------------------------------- #

DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()
DB_HOST: str = os.getenv("DB_HOST", "")
DB_PORT: str = os.getenv("DB_PORT", "5432")
DB_NAME: str = os.getenv("DB_NAME", "postgres")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# --------------------------------------------------------------------------- #
# Servidor
# --------------------------------------------------------------------------- #

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]


def ensure_dirs() -> None:
    """Crea los directorios de datos (imágenes e índices) si no existen."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ayudantes de imagen
# --------------------------------------------------------------------------- #

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Decodifica bytes crudos en una imagen ``PIL.Image`` en modo RGB.

    Fuerza la decodificación completa para que los archivos corruptos fallen
    de inmediato en lugar de hacerlo dentro del modelo.

    Raises
    ------
    ValueError
        Si los bytes no corresponden a una imagen válida.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("El archivo subido no es una imagen válida.") from exc


def list_image_files(directory: str) -> list[str]:
    """
    Lista recursivamente los archivos de imagen de ``directory``.

    Devuelve rutas relativas al directorio, ordenadas alfabéticamente.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    found: list[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                abs_path = os.path.join(root, name)
                found.append(os.path.relpath(abs_path, directory))
    found.sort()
    return found


# --------------------------------------------------------------------------- #
# Metadatos del dataset
# --------------------------------------------------------------------------- #

def load_metadata() -> dict[str, dict]:
    """
    Lee ``metadata.json`` y lo normaliza a ``{nombre_archivo: {"categories": [...], "url": str}}``.

    Acepta también el formato antiguo ``{nombre_archivo: ["cat", ...]}`` para
    mantener compatibilidad con datasets descargados con versiones previas.
    Devuelve un diccionario vacío si el archivo no existe.
    """
    if not os.path.exists(METADATA_PATH):
        return {}
    import json

    with open(METADATA_PATH, encoding="utf-8") as handle:
        raw = json.load(handle)

    normalised: dict[str, dict] = {}
    for filename, value in raw.items():
        if isinstance(value, list):
            normalised[filename] = {"categories": value, "url": ""}
        elif isinstance(value, dict):
            normalised[filename] = {
                "categories": list(value.get("categories", [])),
                "url": value.get("url", "") or "",
            }
    return normalised


# --------------------------------------------------------------------------- #
# Formateo de respuestas
# --------------------------------------------------------------------------- #

def to_image_url(relative_path: str) -> str:
    """Convierte una ruta relativa local en la URL pública servida por la API."""
    clean = relative_path.replace(os.sep, "/").lstrip("/")
    return f"{IMAGE_URL_PREFIX}/{clean}"


def format_results(results: list[dict]) -> list[dict]:
    """
    Transforma los resultados crudos del indexador al esquema de la API.

    Cada entrada de ``results`` trae ``image_path``, ``score``, ``categories``
    y opcionalmente ``image_url`` (URL pública externa, p.ej. el CDN de COCO).
    Si existe URL pública se prefiere; si no, se construye la URL local.
    """
    formatted: list[dict] = []
    for hit in results:
        public_url = (hit.get("image_url") or "").strip()
        formatted.append(
            {
                "image_url": public_url or to_image_url(hit["image_path"]),
                "score": round(float(hit["score"]), 4),
                "categories": list(hit.get("categories") or []),
            }
        )
    return formatted
