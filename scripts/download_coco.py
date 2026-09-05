"""
download_coco.py
================
Descarga un subconjunto balanceado de MS COCO 2017 sin bajar los 18 GB del
dataset completo: obtiene el JSON de anotaciones, elige N imágenes por
categoría y descarga solo esas en paralelo.

Además genera ``data/metadata.json`` con las categorías y la URL pública de
cada imagen, para que el backend pueda devolver enlaces al CDN de COCO en
lugar de servir los archivos localmente (clave para el despliegue).

    python scripts/download_coco.py                       # 100 imgs/categoría, split train
    python scripts/download_coco.py --split val --images-per-cat 40
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import requests
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import utils  # noqa: E402

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_annotations(split: str) -> dict:
    """Descarga el ZIP de anotaciones (~250 MB) y parsea en memoria el JSON del split."""
    print(f"Descargando anotaciones COCO desde {ANNOTATIONS_URL} ...")
    response = requests.get(ANNOTATIONS_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    zip_buffer = io.BytesIO()
    with tqdm(total=total_size, unit="iB", unit_scale=True, desc="annotations.zip") as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            zip_buffer.write(chunk)
            pbar.update(len(chunk))

    member = f"annotations/instances_{split}2017.json"
    print(f"Extrayendo y parseando {member} en memoria ...")
    with zipfile.ZipFile(zip_buffer) as archive:
        with archive.open(member) as handle:
            return json.load(handle)


def download_image(url: str, filepath: str) -> bool:
    """Descarga una imagen; omite las ya existentes para poder reanudar."""
    if os.path.exists(filepath):
        return True
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(filepath, "wb") as handle:
            handle.write(response.content)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga N imágenes por categoría de MS COCO 2017.")
    parser.add_argument(
        "--images-per-cat", type=int, default=100, help="Imágenes por categoría (default 100)."
    )
    parser.add_argument("--workers", type=int, default=20, help="Hilos de descarga concurrentes.")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Split de COCO. 'val' es mucho más ligero de parsear (5k imágenes) pero ofrece menos variedad.",
    )
    args = parser.parse_args()

    utils.ensure_dirs()
    coco = download_annotations(args.split)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_info = {img["id"]: img for img in coco["images"]}
    print(f"{len(categories)} categorías, {len(images_info)} imágenes en el split '{args.split}'.")

    cat_to_images: dict[int, set[int]] = defaultdict(set)
    image_to_cats: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id, img_id = ann["category_id"], ann["image_id"]
        cat_to_images[cat_id].add(img_id)
        name = categories[cat_id]
        if name not in image_to_cats[img_id]:
            image_to_cats[img_id].append(name)

    # Selección determinista: las N primeras imágenes (por id) de cada categoría.
    selected: set[int] = set()
    for cat_id in sorted(categories):
        for img_id in sorted(cat_to_images[cat_id])[: args.images_per_cat]:
            selected.add(img_id)
    print(f"Imágenes únicas seleccionadas: {len(selected)}")

    tasks: list[tuple[str, str]] = []
    metadata: dict[str, dict] = {}
    for img_id in sorted(selected):
        info = images_info.get(img_id)
        if not info or "coco_url" not in info:
            continue
        filename = f"coco_{img_id:012d}.jpg"
        # El CDN de COCO sirve las imágenes por https; se usa esa URL en producción.
        url = info["coco_url"].replace("http://", "https://", 1)
        tasks.append((url, os.path.join(utils.IMAGES_DIR, filename)))
        metadata[filename] = {"categories": image_to_cats[img_id], "url": url}

    print(f"Descargando {len(tasks)} imágenes con {args.workers} hilos ...")
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, url, path): path for url, path in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Imágenes"):
            ok += int(future.result())

    print(f"Descargadas {ok} / {len(tasks)} imágenes en {utils.IMAGES_DIR}")

    with open(utils.METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Metadatos guardados en {utils.METADATA_PATH}")
    print("Siguiente paso: python scripts/build_index.py")


if __name__ == "__main__":
    main()
