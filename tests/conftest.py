"""
Fixtures compartidas.

Las pruebas no cargan el modelo CLIP ni tocan la base de datos: el embedder y
el indexador se sustituyen por dobles deterministas para que la suite corra
en segundos, sin GPU ni credenciales. Si ``torch`` u ``open_clip`` no están
instalados se inyectan módulos mínimos para poder importar ``backend``.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("INDEX_BACKEND", "pgvector")
os.environ.setdefault("CORS_ORIGINS", "*")


def _stub_ml_modules() -> None:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda fn: fn)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=None)
    torch.nn = types.SimpleNamespace(Module=object)
    torch.Tensor = object
    torch.stack = lambda *_a, **_k: None
    sys.modules["torch"] = torch

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = lambda *_a, **_k: (None, None, None)
    open_clip.get_tokenizer = lambda *_a, **_k: None
    sys.modules["open_clip"] = open_clip


_stub_ml_modules()

from backend import embedder, indexer  # noqa: E402
from backend.main import app  # noqa: E402

DIM = 512


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM).astype("float32")
    return vec / np.linalg.norm(vec)


FAKE_HITS = [
    {
        "image_path": "coco_000000000001.jpg",
        "image_url": "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000001.jpg",
        "categories": ["cat", "couch"],
        "score": 0.3123456,
    },
    {
        "image_path": "sub\\local_image.jpg",
        "image_url": "",
        "categories": [],
        "score": 0.25,
    },
]


@pytest.fixture()
def ready_backend(monkeypatch):
    """Modelo e índice 'cargados' con dobles deterministas."""
    monkeypatch.setattr(embedder, "load_model", lambda *_a, **_k: None)
    monkeypatch.setattr(embedder, "is_ready", lambda: True)
    monkeypatch.setattr(embedder, "get_text_embedding", lambda text: _unit_vector(len(text)))
    monkeypatch.setattr(embedder, "get_image_embedding", lambda image: _unit_vector(image.size[0]))

    calls: dict = {}

    def fake_search(vector, top_k=6):
        calls["top_k"] = top_k
        calls["vector"] = vector
        return FAKE_HITS[:top_k]

    monkeypatch.setattr(indexer, "load_index", lambda: None)
    monkeypatch.setattr(indexer, "_ready", True)
    monkeypatch.setattr(indexer, "search", fake_search)
    monkeypatch.setattr(indexer, "index_size", lambda: len(FAKE_HITS))
    monkeypatch.setattr(
        indexer,
        "stats",
        lambda: {
            "backend": indexer.BACKEND,
            "total_images": 2,
            "categories": [{"name": "cat", "count": 1}, {"name": "couch", "count": 1}],
        },
    )
    return calls


@pytest.fixture()
def client(ready_backend):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client_not_ready(monkeypatch):
    """Servidor arrancado pero sin índice (modo degradado)."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(embedder, "load_model", lambda *_a, **_k: None)
    monkeypatch.setattr(embedder, "is_ready", lambda: True)
    monkeypatch.setattr(indexer, "load_index", lambda: (_ for _ in ()).throw(RuntimeError("sin BD")))
    monkeypatch.setattr(indexer, "_ready", False)
    with TestClient(app) as test_client:
        yield test_client
