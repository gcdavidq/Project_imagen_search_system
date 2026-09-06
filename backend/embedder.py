
from __future__ import annotations

import logging
import threading

import numpy as np
import open_clip
import torch
from PIL import Image

from . import utils

logger = logging.getLogger(__name__)


_model: torch.nn.Module | None = None
_preprocess = None
_tokenizer = None
_device: str = "cpu"


_load_lock = threading.Lock()


def _select_device() -> str:
    """
    Selecciona el dispositivo de cómputo óptimo disponible en el sistema.

    Evalúa la disponibilidad de CUDA (GPU NVIDIA), MPS (GPU Apple Silicon)
    y regresa a CPU como opción por defecto.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    str
        Identificador del dispositivo PyTorch: ``'cuda'``, ``'mps'`` o ``'cpu'``.
    """
    if torch.cuda.is_available():
        return "cuda"
    # GPUs de Apple Silicon.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = utils.MODEL_NAME, pretrained: str = utils.PRETRAINED) -> None:
    """
    Carga el modelo CLIP, el preprocesador de imágenes y el tokenizador de texto
    en variables globales del módulo (caché singleton).

    Es seguro invocarlo múltiples veces: las llamadas posteriores a la primera
    son no-ops gracias al patrón double-checked locking con ``threading.Lock``.

    Parameters
    ----------
    model_name : str, opcional
        Nombre de la arquitectura del modelo a cargar mediante ``open_clip``.
        Por defecto usa ``utils.MODEL_NAME`` (``'xlm-roberta-base-ViT-B-32'``).
    pretrained : str, opcional
        Identificador de los pesos preentrenados a usar.
        Por defecto usa ``utils.PRETRAINED`` (``'laion5b_s13b_b90k'``).

    Returns
    -------
    None

    Raises
    ------
    Exception
        Cualquier error lanzado por ``open_clip.create_model_and_transforms``
        (modelo no encontrado, pesos no disponibles, error de descarga, etc.)
        se propaga sin capturarse.

    Notes
    -----
    - El modelo se mueve al dispositivo seleccionado por ``_select_device()``
      y se pone en modo evaluación (``model.eval()``).
    - Las variables globales ``_model``, ``_preprocess``, ``_tokenizer`` y
      ``_device`` son actualizadas como efecto secundario.
    """
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return

    with _load_lock:
        # Volver a comprobar dentro del bloqueo en caso de que otro hilo haya ganado la carrera.
        if _model is not None:
            return

        _device = _select_device()
        logger.info("Cargando el modelo CLIP '%s' (pretrained=%s) en %s ...",
                    model_name, pretrained, _device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(_device)
        model.eval()

        _model = model
        _preprocess = preprocess
        _tokenizer = open_clip.get_tokenizer(model_name)

        logger.info("Modelo CLIP cargado. Dimensión de incrustación: %d", get_embedding_dim())


def is_ready() -> bool:
    """
    Indica si el modelo CLIP ya fue cargado y está disponible para inferencia.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    bool
        ``True`` si ``_model`` es distinto de ``None`` (modelo cargado);
        ``False`` en caso contrario.
    """
    return _model is not None


def _ensure_loaded() -> None:
    """
    Garantiza que el modelo CLIP esté cargado antes de usarlo (carga perezosa).

    Si ``load_model()`` no fue llamado explícitamente durante el arranque de la
    aplicación, este helper lo invoca de forma automática la primera vez que
    se solicite un embedding.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    None

    Notes
    -----
    Delega toda la lógica de inicialización a :func:`load_model`, que maneja
    la concurrencia mediante ``threading.Lock``.
    """
    if _model is None:
        load_model()


def get_embedding_dim() -> int:
    """
    Devuelve la dimensionalidad del espacio de embedding del modelo CLIP cargado.

    Inspecciona el modelo en el siguiente orden de prioridad:
    1. ``model.visual.output_dim`` (disponible en la mayoría de arquitecturas ViT).
    2. ``model.text_projection.shape[1]`` (fallback para modelos con proyección de texto).
    3. ``512`` como valor por defecto para ViT-B/32.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    int
        Número de dimensiones del vector de embedding (p.ej. ``512`` para ViT-B/32).

    Notes
    -----
    Llama a ``_ensure_loaded()`` internamente; no es necesario llamar
    a ``load_model()`` antes de invocar esta función.
    """
    _ensure_loaded()
    if hasattr(_model, "visual") and hasattr(_model.visual, "output_dim"):
        return int(_model.visual.output_dim)
    if hasattr(_model, "text_projection") and _model.text_projection is not None:
        return int(_model.text_projection.shape[1])
    return 512


def _l2_normalise(features: torch.Tensor) -> torch.Tensor:
    """
    Aplica normalización L2 a un lote de vectores de características.

    Divide cada vector por su norma Euclidiana a lo largo de la última
    dimensión, produciendo vectores unitarios aptos para similitud del coseno.

    Parameters
    ----------
    features : torch.Tensor
        Tensor de forma ``(batch_size, embedding_dim)`` con los vectores
        de características crudas generados por el encoder CLIP.

    Returns
    -------
    torch.Tensor
        Tensor de la misma forma que ``features``, donde cada fila tiene
        norma L2 igual a 1.0.
    """
    return features / features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    """
    Genera el embedding CLIP de un texto de consulta, normalizado L2.

    Tokeniza la cadena de entrada, la pasa por el encoder de texto del
    modelo CLIP y normaliza el vector resultante para usarlo en búsqueda
    por similitud del coseno.

    Parameters
    ----------
    text : str
        Texto de consulta en cualquier idioma soportado por el tokenizador
        del modelo (multilingüe para ``xlm-roberta-base-ViT-B-32``).

    Returns
    -------
    np.ndarray
        Arreglo 1-D de tipo ``float32`` con forma ``(embedding_dim,)``
        (p.ej. ``(512,)`` para ViT-B/32), listo para comparación vectorial.

    Notes
    -----
    - Ejecutado bajo ``@torch.no_grad()`` para ahorrar memoria y acelerar
      la inferencia.
    - Llama a ``_ensure_loaded()`` internamente antes de tokenizar.
    """
    _ensure_loaded()
    tokens = _tokenizer([text]).to(_device)          # type: ignore[misc]
    features = _model.encode_text(tokens)            # type: ignore[union-attr]
    features = _l2_normalise(features)
    return features.cpu().numpy().astype("float32")[0]


@torch.no_grad()
def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Genera el embedding CLIP de una imagen PIL, normalizado L2.

    Aplica la transformación de preprocesamiento del modelo (redimensionar,
    recortar y normalizar píxeles), pasa el tensor por el encoder visual
    de CLIP y normaliza el vector resultante para búsqueda por similitud.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagen de consulta en modo RGB. Se recomienda convertirla a RGB
        antes de invocar esta función (ver :func:`utils.load_image_from_bytes`).

    Returns
    -------
    np.ndarray
        Arreglo 1-D de tipo ``float32`` con forma ``(embedding_dim,)``
        (p.ej. ``(512,)`` para ViT-B/32), listo para comparación vectorial.

    Notes
    -----
    - Ejecutado bajo ``@torch.no_grad()`` para ahorrar memoria y acelerar
      la inferencia.
    - Llama a ``_ensure_loaded()`` internamente antes de preprocesar.
    """
    _ensure_loaded()
    tensor = _preprocess(image).unsqueeze(0).to(_device)   # type: ignore[misc]
    features = _model.encode_image(tensor)                 # type: ignore[union-attr]
    features = _l2_normalise(features)
    return features.cpu().numpy().astype("float32")[0]


@torch.no_grad()
def get_image_embeddings(images: list[Image.Image]) -> list[np.ndarray]:
    """
    Genera embeddings CLIP normalizados L2 para un lote de imágenes PIL.

    Procesar varias imágenes en una sola pasada es mucho más rápido que
    llamar a :func:`get_image_embedding` una por una (usado por
    ``scripts/build_index.py``).

    Returns
    -------
    List[np.ndarray]
        Un vector ``float32`` de forma ``(embedding_dim,)`` por imagen, en el
        mismo orden de entrada.
    """
    _ensure_loaded()
    if not images:
        return []
    batch = torch.stack([_preprocess(img) for img in images]).to(_device)  # type: ignore[misc]
    features = _model.encode_image(batch)                                  # type: ignore[union-attr]
    features = _l2_normalise(features)
    return list(features.cpu().numpy().astype("float32"))
