"""
Pruebas de la construcción de URLs públicas de MS COCO.

Regresión importante: ``images.cocodataset.org`` es un alias de un bucket S3
cuyo certificado solo cubre ``s3.amazonaws.com`` y ``*.s3.amazonaws.com``.
Usar ``https://images.cocodataset.org/...`` rompe la descarga (verificación de
certificado) y también la carga de imágenes en el navegador, mientras que
``http://`` es bloqueado como contenido mixto por un frontend servido con TLS.
La forma path-style es la única que funciona en los tres contextos.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from download_coco import to_public_https_url  # noqa: E402

BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        # Lo que trae realmente el JSON de anotaciones de COCO.
        (
            "http://images.cocodataset.org/train2017/000000000009.jpg",
            f"{BUCKET}/train2017/000000000009.jpg",
        ),
        # Si COCO empezara a publicar https, el resultado debe ser el mismo.
        (
            "https://images.cocodataset.org/val2017/000000000139.jpg",
            f"{BUCKET}/val2017/000000000139.jpg",
        ),
        # Idempotente: una URL ya convertida no se toca.
        (
            f"{BUCKET}/train2017/000000000009.jpg",
            f"{BUCKET}/train2017/000000000009.jpg",
        ),
        # Otros hosts de S3 se mantienen, solo se fuerza TLS.
        (
            "http://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/x.jpg",
            "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/x.jpg",
        ),
        # Entrada vacía o sin host: se devuelve tal cual, sin romper.
        ("", ""),
        ("no-es-una-url", "no-es-una-url"),
    ],
)
def test_to_public_https_url(entrada, esperado):
    assert to_public_https_url(entrada) == esperado


def test_resultado_siempre_es_https_para_urls_de_coco():
    url = to_public_https_url("http://images.cocodataset.org/train2017/000000000009.jpg")
    assert url.startswith("https://")
    # El host debe estar cubierto por el certificado de S3.
    assert url.split("/")[2] == "s3.amazonaws.com"
