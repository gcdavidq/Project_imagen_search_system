
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image

import open_clip

from . import utils

logger = logging.getLogger(__name__)


_model: Optional[torch.nn.Module] = None
_preprocess = None
_tokenizer = None
_device: str = "cpu"


_load_lock = threading.Lock()


def _select_device() -> str:
    """Seleccionar el mejor dispositivo torch disponible."""
    if torch.cuda.is_available():
        return "cuda"
    # GPUs de Apple Silicon.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = utils.MODEL_NAME, pretrained: str = utils.PRETRAINED) -> None:
    """
    Cargar el modelo CLIP, la transformación de preprocesamiento y el tokenizador en la caché
    del módulo. Seguro para llamar múltiples veces -- las llamadas subsecuentes no hacen nada.
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
    """Devolver ``True`` una vez que el modelo esté cargado y listo para inferencia."""
    return _model is not None


def _ensure_loaded() -> None:
    """Cargar el modelo de forma perezosa (lazy) si :func:`load_model` no fue llamado por adelantado."""
    if _model is None:
        load_model()


def get_embedding_dim() -> int:
    """Devolver la dimensionalidad del espacio de incrustación de CLIP (512 para ViT-B-32)."""
    _ensure_loaded()
    if hasattr(_model, "visual") and hasattr(_model.visual, "output_dim"):
        return int(_model.visual.output_dim)
    if hasattr(_model, "text_projection") and _model.text_projection is not None:
        return int(_model.text_projection.shape[1])
    return 512


def _l2_normalise(features: torch.Tensor) -> torch.Tensor:
    """Normalizar L2 un lote de vectores de características a lo largo de la última dimensión."""
    return features / features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    """
    Tokenizar ``text`` y devolver su incrustación CLIP normalizada L2.

    Returns
    -------
    np.ndarray
        Un arreglo 1-D ``float32`` de forma ``(embedding_dim,)``.
    """
    _ensure_loaded()
    tokens = _tokenizer([text]).to(_device)          # type: ignore[misc]
    features = _model.encode_text(tokens)            # type: ignore[union-attr]
    features = _l2_normalise(features)
    return features.cpu().numpy().astype("float32")[0]


@torch.no_grad()
def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Preprocesar una imagen PIL y devolver su incrustación CLIP normalizada L2.

    Returns
    -------
    np.ndarray
        Un arreglo 1-D ``float32`` de forma ``(embedding_dim,)``.
    """
    _ensure_loaded()
    tensor = _preprocess(image).unsqueeze(0).to(_device)   # type: ignore[misc]
    features = _model.encode_image(tensor)                 # type: ignore[union-attr]
    features = _l2_normalise(features)
    return features.cpu().numpy().astype("float32")[0]
