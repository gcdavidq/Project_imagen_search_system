"""
utils.py
========
Configuración centralizada (leída desde el entorno / .env) y pequeños ayudantes
reutilizables compartidos en el backend: carga de imágenes, validación y
formateo de respuestas.


"""

from __future__ import annotations

import io
import logging
import os
from typing import List, Dict

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

# Cargar variables desde un archivo .env local si existe.
load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Rutas y configuración
# --------------------------------------------------------------------------- #

# La raíz del proyecto es el padre del paquete ``backend`` (es decir, image_search_system/).
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ``DATA_DIR`` se puede sobreescribir a través del entorno; todo lo demás se
# deriva de él para que el diseño se mantenga consistente.
DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "embeddings")

INDEX_PATH: str = os.path.join(EMBEDDINGS_DIR, "index.faiss")
PATHS_PATH: str = os.path.join(EMBEDDINGS_DIR, "image_paths.json")


# Selección del modelo CLIP.
MODEL_NAME: str = os.getenv("MODEL_NAME", "ViT-B-32")
PRETRAINED: str = os.getenv("PRETRAINED", "openai")

# Número predeterminado de resultados devueltos por una búsqueda.
TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "6"))

# Prefijo de URL pública bajo el cual se sirven las imágenes del conjunto de datos (ver montaje en main.py).
IMAGE_URL_PREFIX: str = "/images"

# Tipos de contenido de imagen permitidos para el endpoint de carga.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}

# Configuración de la base de datos
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_SECRET_KEY", "")
DB_HOST: str = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT: str = os.getenv("SUPABASE_DB_PORT", "5432")
DB_NAME: str = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER: str = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("SUPABASE_DB_PASSWORD", "")


def ensure_dirs() -> None:
    """Crear los directorios de datos si aún no existen."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ayudantes de imagen
# --------------------------------------------------------------------------- #

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Decodificar bytes crudos en una imagen RGB :class:`PIL.Image.Image`.

    Raises
    ------
    ValueError
        Si los bytes no contienen una imagen válida y decodificable.
    """
    try:
        image = Image.open(io.BytesIO(data))
        # ``load`` fuerza la decodificación para que los archivos malformados fallen aquí en lugar de más tarde.
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("El archivo subido no es una imagen válida.") from exc


def list_image_files(directory: str) -> List[str]:
    """
    Devolver una lista ordenada de rutas de archivos de imagen (relativas a ``directory``) encontradas
    recursivamente bajo ``directory``.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    found: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                abs_path = os.path.join(root, name)
                found.append(os.path.relpath(abs_path, directory))
    found.sort()
    return found


# --------------------------------------------------------------------------- #
# Formateo de respuestas
# --------------------------------------------------------------------------- #

def to_image_url(relative_path: str) -> str:
    """Convertir una ruta relativa a ``IMAGES_DIR`` en una URL de imagen pública."""
    # Normalizar los separadores de Windows para que las URLs estén siempre basadas en barras diagonales.
    clean = relative_path.replace(os.sep, "/").lstrip("/")
    return f"{IMAGE_URL_PREFIX}/{clean}"


def format_results(results: List[Dict]) -> List[Dict]:
    """
    Convertir las coincidencias brutas del indexador ``[{"image_path", "score", "categories"}]`` en la forma de
    respuesta de la API esperada por el frontend.
    """
    formatted: List[Dict] = []
    for hit in results:
        formatted_hit = {
            "image_url": to_image_url(hit["image_path"]),
            "score": round(float(hit["score"]), 4),
        }
        if "categories" in hit:
            formatted_hit["categories"] = hit["categories"]
        formatted.append(formatted_hit)
    return formatted
