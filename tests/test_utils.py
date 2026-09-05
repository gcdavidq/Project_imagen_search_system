"""Pruebas unitarias de los ayudantes de configuración y formateo."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from backend import utils


def test_to_image_url_normalises_windows_separators():
    assert utils.to_image_url("carpeta\\foto.jpg") == "/images/carpeta/foto.jpg"
    assert utils.to_image_url("/ya/limpia.png") == "/images/ya/limpia.png"


def test_format_results_prefers_public_url_and_rounds_score():
    hits = [
        {"image_path": "a.jpg", "image_url": "https://cdn/a.jpg", "score": 0.123456, "categories": ["dog"]},
        {"image_path": "b.jpg", "score": "0.5"},
    ]
    out = utils.format_results(hits)
    assert out[0] == {"image_url": "https://cdn/a.jpg", "score": 0.1235, "categories": ["dog"]}
    assert out[1] == {"image_url": "/images/b.jpg", "score": 0.5, "categories": []}


def test_load_image_from_bytes_converts_to_rgb():
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8)).save(buffer, format="PNG")
    image = utils.load_image_from_bytes(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_load_image_from_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        utils.load_image_from_bytes(b"definitivamente no es una imagen")


def test_list_image_files_is_recursive_and_sorted(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["b.JPG", "a.png", "sub/c.webp", "ignorar.txt"]:
        (tmp_path / name).write_bytes(b"")
    found = utils.list_image_files(str(tmp_path))
    assert [p.replace("\\", "/") for p in found] == ["a.png", "b.JPG", "sub/c.webp"]


def test_load_metadata_accepts_legacy_and_new_formats(tmp_path, monkeypatch):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "old.jpg": ["cat"],
                "new.jpg": {"categories": ["dog"], "url": "https://cdn/new.jpg"},
                "partial.jpg": {"categories": ["bird"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "METADATA_PATH", str(path))

    meta = utils.load_metadata()
    assert meta["old.jpg"] == {"categories": ["cat"], "url": ""}
    assert meta["new.jpg"] == {"categories": ["dog"], "url": "https://cdn/new.jpg"}
    assert meta["partial.jpg"] == {"categories": ["bird"], "url": ""}


def test_load_metadata_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "METADATA_PATH", str(tmp_path / "nope.json"))
    assert utils.load_metadata() == {}
