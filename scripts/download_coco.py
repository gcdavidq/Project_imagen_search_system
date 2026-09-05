"""
download_coco.py
================
Descarga un subconjunto balanceado de MS COCO 2017 sin bajar los 18 GB del
dataset completo: obtiene el JSON de anotaciones, elige N imágenes por
categoría y descarga solo esas en paralelo.

Además genera ``data/metadata.json`` con las categorías y la URL pública de
cada imagen, para que el backend pueda devolver enlaces al almacenamiento de
COCO en lugar de servir los archivos localmente (clave para el despliegue).

    python scripts/download_coco.py                       # 100 imgs/categoría, split train
    python scripts/download_coco.py --split val --images-per-cat 40
    python scripts/download_coco.py --reuse-metadata      # reintenta la descarga sin volver
                                                          # a bajar las anotaciones (250 MB)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from urllib.parse import urlparse

import requests
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import utils  # noqa: E402

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def to_public_https_url(coco_url: str) -> str:
    """
    Devuelve una URL con TLS válido para una imagen de COCO.

    ``images.cocodataset.org`` es un alias (CNAME) de un bucket de Amazon S3
    cuyo certificado solo cubre ``s3.amazonaws.com`` y ``*.s3.amazonaws.com``.
    Por eso ``https://images.cocodataset.org/...`` falla la verificación del
    certificado en **cualquier** cliente, navegadores incluidos, y ``http://``
    no sirve porque un frontend servido por HTTPS bloquea el contenido mixto.

    La forma *path-style* ``https://s3.amazonaws.com/<bucket>/<clave>`` entrega
    exactamente los mismos bytes con un certificado válido.

    >>> to_public_https_url("http://images.cocodataset.org/train2017/000000000009.jpg")
    'https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000009.jpg'
    """
    parsed = urlparse(coco_url)
    if not parsed.netloc:
        return coco_url
    if parsed.netloc.endswith("amazonaws.com"):
        return coco_url.replace("http://", "https://", 1)
    return f"https://s3.amazonaws.com/{parsed.netloc}{parsed.path}"


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


def download_image(url: str, filepath: str) -> tuple[bool, str]:
    """
    Descarga una imagen; omite las ya existentes para poder reanudar.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` si se descargó o ya existía. En caso de fallo,
        ``(False, motivo)`` con una descripción corta para poder agregar los
        errores por tipo, en vez de silenciarlos.
    """
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return True, ""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if not response.content:
            return False, "respuesta vacía"
        with open(filepath, "wb") as handle:
            handle.write(response.content)
        return True, ""
    except requests.exceptions.SSLError as exc:
        return False, f"SSL: {str(exc)[:90]}"
    except requests.exceptions.HTTPError as exc:
        return False, f"HTTP {exc.response.status_code if exc.response is not None else '?'}"
    except requests.exceptions.RequestException as exc:
        return False, f"{type(exc).__name__}"
    except OSError as exc:
        return False, f"disco: {exc.strerror or exc}"


def build_tasks_from_coco(split: str, images_per_cat: int) -> tuple[list[tuple[str, str]], dict]:
    """Descarga las anotaciones y devuelve (tareas de descarga, metadatos)."""
    coco = download_annotations(split)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_info = {img["id"]: img for img in coco["images"]}
    print(f"{len(categories)} categorías, {len(images_info)} imágenes en el split '{split}'.")

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
        for img_id in sorted(cat_to_images[cat_id])[:images_per_cat]:
            selected.add(img_id)
    print(f"Imágenes únicas seleccionadas: {len(selected)}")

    tasks: list[tuple[str, str]] = []
    metadata: dict[str, dict] = {}
    for img_id in sorted(selected):
        info = images_info.get(img_id)
        if not info or "coco_url" not in info:
            continue
        filename = f"coco_{img_id:012d}.jpg"
        url = to_public_https_url(info["coco_url"])
        tasks.append((url, os.path.join(utils.IMAGES_DIR, filename)))
        metadata[filename] = {"categories": image_to_cats[img_id], "url": url}
    return tasks, metadata


def build_tasks_from_metadata() -> tuple[list[tuple[str, str]], dict]:
    """Reconstruye las tareas desde un ``metadata.json`` existente, sin red."""
    metadata = utils.load_metadata()
    if not metadata:
        print(f"[ERROR] No hay metadatos en {utils.METADATA_PATH}.")
        print("Ejecuta el script sin --reuse-metadata para generarlos.")
        sys.exit(1)

    print(f"Reutilizando {len(metadata)} entradas de {utils.METADATA_PATH}")
    tasks: list[tuple[str, str]] = []
    repaired = 0
    for filename, meta in metadata.items():
        url = to_public_https_url(meta.get("url", ""))
        if url and url != meta.get("url"):
            meta["url"] = url
            repaired += 1
        if url:
            tasks.append((url, os.path.join(utils.IMAGES_DIR, filename)))
    if repaired:
        print(f"Se corrigieron {repaired} URLs para que validen por HTTPS.")
    return tasks, metadata


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
    parser.add_argument(
        "--reuse-metadata",
        action="store_true",
        help="Reintenta la descarga usando el metadata.json existente, sin volver a bajar las anotaciones.",
    )
    args = parser.parse_args()

    utils.ensure_dirs()

    if args.reuse_metadata:
        tasks, metadata = build_tasks_from_metadata()
    else:
        tasks, metadata = build_tasks_from_coco(args.split, args.images_per_cat)

    # Guardar los metadatos antes de descargar: son lo que build_index.py necesita
    # y no dependen de que la descarga termine.
    with open(utils.METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Metadatos guardados en {utils.METADATA_PATH}")

    print(f"Descargando {len(tasks)} imágenes con {args.workers} hilos ...")
    ok = 0
    errors: Counter = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, url, path): path for url, path in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Imágenes"):
            success, reason = future.result()
            if success:
                ok += 1
            else:
                errors[reason] += 1

    print(f"\nDescargadas {ok} / {len(tasks)} imágenes en {utils.IMAGES_DIR}")

    if errors:
        print("\nFallos por motivo:")
        for reason, count in errors.most_common(5):
            print(f"  {count:>6} × {reason}")
        if ok == 0:
            print(
                "\nNinguna imagen se descargó. Comprueba tu conexión y que "
                f"{tasks[0][0] if tasks else 'la URL'} sea accesible desde tu red."
            )
            sys.exit(1)

    print("\nSiguiente paso: python scripts/build_index.py")


if __name__ == "__main__":
    main()
