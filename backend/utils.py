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


# Selección del modelo CLIP multilingüe (OpenCLIP).
# Modelo: xlm-roberta-base-ViT-B-32, pesos: laion5b_s13b_b90k
# Soporta búsquedas en múltiples idiomas gracias al encoder de texto multilingüe.
MODEL_NAME: str = os.getenv("MODEL_NAME", "xlm-roberta-base-ViT-B-32")
PRETRAINED: str = os.getenv("PRETRAINED", "laion5b_s13b_b90k")

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
    """
    Crea los directorios de datos del proyecto si aún no existen.

    Crea de forma recursiva (``os.makedirs(..., exist_ok=True)``) los
    directorios ``IMAGES_DIR`` y ``EMBEDDINGS_DIR`` definidos en este módulo.
    No hace nada si los directorios ya existen.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    None
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ayudantes de imagen
# --------------------------------------------------------------------------- #

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Decodifica bytes crudos en una imagen RGB ``PIL.Image.Image``.

    Abre la imagen desde el buffer de bytes, fuerza la decodificación
    completa (para detectar archivos corruptos de inmediato) y convierte
    el resultado al modo ``RGB`` para compatibilidad con el preprocesador CLIP.

    Parameters
    ----------
    data : bytes
        Contenido binario de un archivo de imagen (JPG, PNG, WebP, BMP, etc.).
        Obtenido típicamente de ``await upload_file.read()`` en FastAPI.

    Returns
    -------
    PIL.Image.Image
        Imagen decodificada en modo ``RGB``, lista para ser pasada al
        preprocesador de CLIP (``embedder.get_image_embedding``).

    Raises
    ------
    ValueError
        Si los bytes no contienen una imagen válida o decodificable
        (archivo corrupto, formato no soportado, bytes truncados, etc.).
        Envuelve la excepción original de Pillow como causa.
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
    Lista recursivamente todos los archivos de imagen dentro de un directorio.

    Recorre el árbol de directorios con ``os.walk`` y filtra los archivos
    cuya extensión sea ``.jpg``, ``.jpeg``, ``.png``, ``.webp`` o ``.bmp``.
    Los resultados se devuelven como rutas relativas al directorio raíz,
    ordenadas alfabéticamente.

    Parameters
    ----------
    directory : str
        Ruta absoluta o relativa al directorio raíz donde buscar imágenes.
        Generalmente es ``utils.IMAGES_DIR``.

    Returns
    -------
    List[str]
        Lista ordenada de rutas relativas (respecto a ``directory``) de
        todos los archivos de imagen encontrados. Puede ser lista vacía
        si no se encuentra ninguna imagen.

    Notes
    -----
    - La búsqueda es **recursiva**: incluye imágenes en subdirectorios.
    - La comparación de extensiones es insensible a mayúsculas/minúsculas
      (usa ``.lower()``).
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
    """
    Convierte una ruta relativa de imagen en una URL pública servida por la API.

    Combina el prefijo de URL configurado (``IMAGE_URL_PREFIX``, por defecto
    ``/images``) con la ruta relativa normalizada (separadores de Windows
    convertidos a ``/``).

    Parameters
    ----------
    relative_path : str
        Ruta del archivo relativa a ``IMAGES_DIR``.
        Ejemplo: ``"gatos\\siames.jpg"`` (Windows) o ``"gatos/siames.jpg"``.

    Returns
    -------
    str
        URL pública de la imagen lista para incrustar en respuestas JSON.
        Ejemplo: ``"/images/gatos/siames.jpg"``.
    """
    # Normalizar los separadores de Windows para que las URLs estén siempre basadas en barras diagonales.
    clean = relative_path.replace(os.sep, "/").lstrip("/")
    return f"{IMAGE_URL_PREFIX}/{clean}"


def format_results(results: List[Dict]) -> List[Dict]:
    """
    Transforma la lista de resultados crudos del indexador al formato de respuesta de la API.

    Convierte cada entrada del indexador (que usa rutas de archivo locales)
    en un diccionario con URLs públicas y puntuaciones redondeadas,
    compatible con el esquema Pydantic ``SearchResult`` del frontend.

    Parameters
    ----------
    results : List[Dict]
        Lista de diccionarios retornados por ``indexer.search()``. Cada
        elemento debe contener al menos:

        - ``"image_path"`` (str): ruta relativa de la imagen en ``IMAGES_DIR``.
        - ``"score"`` (float): puntuación de similitud del coseno.
        - ``"categories"`` (list, opcional): categorías de la imagen.

    Returns
    -------
    List[Dict]
        Lista de diccionarios transformados. Cada elemento contiene:

        - ``"image_url"`` (str): URL pública construida por ``to_image_url()``.
        - ``"score"`` (float): puntuación redondeada a 4 decimales.
        - ``"categories"`` (list, opcional): incluido solo si estaba presente
          en el resultado de entrada.

    Notes
    -----
    - La puntuación se redondea a 4 decimales para evitar ruido de punto
      flotante en la respuesta JSON.
    - Los separadores de ruta de Windows son normalizados automáticamente
      por ``to_image_url()``.
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
