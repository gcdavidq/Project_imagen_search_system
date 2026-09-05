"""Pruebas de los endpoints HTTP con el modelo y el índice sustituidos por dobles."""

from __future__ import annotations

import io

from PIL import Image


def _png_bytes(size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color=(120, 40, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Sistema
# --------------------------------------------------------------------------- #

def test_root_lists_docs(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


def test_health_reports_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["index_loaded"] is True
    assert body["index_size"] == 2
    assert body["backend"] in {"pgvector", "faiss"}


def test_health_reports_degraded_mode(client_not_ready):
    body = client_not_ready.get("/health").json()
    assert body["index_loaded"] is False
    assert body["index_size"] == 0


def test_stats(client):
    body = client.get("/stats").json()
    assert body["total_images"] == 2
    assert body["categories"][0] == {"name": "cat", "count": 1}


# --------------------------------------------------------------------------- #
# Texto → imagen
# --------------------------------------------------------------------------- #

def test_text_search_returns_formatted_results(client, ready_backend):
    response = client.post("/search/text", json={"query": "un gato en un sofá", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["count"] == 2
    assert body["took_ms"] >= 0
    assert ready_backend["top_k"] == 2

    first, second = body["results"]
    # URL pública preferida cuando existe; score redondeado a 4 decimales.
    assert first["image_url"].startswith("https://images.cocodataset.org/")
    assert first["score"] == 0.3123
    assert first["categories"] == ["cat", "couch"]
    # Sin URL pública: se construye la URL local con separadores normalizados.
    assert second["image_url"] == "/images/sub/local_image.jpg"
    assert second["categories"] == []


def test_text_search_uses_default_top_k(client, ready_backend):
    client.post("/search/text", json={"query": "perro"})
    assert ready_backend["top_k"] == 6


def test_text_search_rejects_blank_query(client):
    assert client.post("/search/text", json={"query": "   "}).status_code == 400
    assert client.post("/search/text", json={"query": ""}).status_code == 422


def test_text_search_validates_top_k_range(client):
    assert client.post("/search/text", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search/text", json={"query": "x", "top_k": 51}).status_code == 422


def test_text_search_503_when_index_not_ready(client_not_ready):
    response = client_not_ready.post("/search/text", json={"query": "gato"})
    assert response.status_code == 503
    assert "índice" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Imagen → imagen
# --------------------------------------------------------------------------- #

def test_image_search_accepts_png(client, ready_backend):
    response = client.post(
        "/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert ready_backend["top_k"] == 1


def test_image_search_rejects_unsupported_type(client):
    response = client.post("/search/image", files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")})
    assert response.status_code == 400


def test_image_search_rejects_empty_file(client):
    response = client.post("/search/image", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_image_search_rejects_corrupt_bytes(client):
    response = client.post("/search/image", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert "imagen válida" in response.json()["detail"]


def test_image_search_rejects_oversized_file(client):
    from backend.routes import image_search

    big = b"\x00" * (image_search.MAX_UPLOAD_BYTES + 1)
    response = client.post("/search/image", files={"file": ("big.png", big, "image/png")})
    assert response.status_code == 400
    assert "10 MB" in response.json()["detail"]


def test_image_search_503_when_index_not_ready(client_not_ready):
    response = client_not_ready.post("/search/image", files={"file": ("q.png", _png_bytes(), "image/png")})
    assert response.status_code == 503
