"""
build_index.py
==============
Genera los embeddings CLIP de todas las imágenes de ``data/images/`` y
construye el índice vectorial en el backend activo (``INDEX_BACKEND``):

    python scripts/build_index.py                  # backend según .env
    python scripts/build_index.py --backend faiss  # fuerza índice local offline
    python scripts/build_index.py --images-dir ruta/a/imagenes --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import sys

# Hacer importable el paquete ``backend`` al ejecutar el script directamente.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye el índice vectorial a partir de data/images/.")
    parser.add_argument(
        "--images-dir", default=None, help="Carpeta con las imágenes (por defecto DATA_DIR/images)."
    )
    parser.add_argument(
        "--backend",
        choices=["pgvector", "faiss"],
        default=None,
        help="Sobrescribe INDEX_BACKEND solo para esta ejecución.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Imágenes por lote de inferencia.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Procesa solo las primeras N imágenes (pruebas)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend:
        # Debe fijarse antes de importar ``backend``, que lee la variable al cargar.
        os.environ["INDEX_BACKEND"] = args.backend

    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    from backend import embedder, indexer, utils

    images_dir = args.images_dir or utils.IMAGES_DIR
    image_paths = utils.list_image_files(images_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print(f"[ERROR] No se encontraron imágenes en '{images_dir}'.")
        print("Ejecuta primero: python scripts/download_coco.py")
        sys.exit(1)

    print(f"Encontradas {len(image_paths)} imágenes en {images_dir}.")
    print(f"Backend: {indexer.BACKEND} | Modelo: {utils.MODEL_NAME}/{utils.PRETRAINED}")
    print("Cargando modelo CLIP ...")
    embedder.load_model(utils.MODEL_NAME, utils.PRETRAINED)

    embeddings: list[np.ndarray] = []
    kept_paths: list[str] = []

    batch_images: list[Image.Image] = []
    batch_paths: list[str] = []

    def flush() -> None:
        if not batch_images:
            return
        vectors = embedder.get_image_embeddings(batch_images)
        embeddings.extend(vectors)
        kept_paths.extend(batch_paths)
        batch_images.clear()
        batch_paths.clear()

    for rel_path in tqdm(image_paths, desc="Generando embeddings", unit="img"):
        abs_path = os.path.join(images_dir, rel_path)
        try:
            with Image.open(abs_path) as img:
                batch_images.append(img.convert("RGB"))
            batch_paths.append(rel_path)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  omitida {rel_path}: {exc}")
            continue
        if len(batch_images) >= args.batch_size:
            flush()
    flush()

    if not embeddings:
        print("[ERROR] Ninguna imagen pudo procesarse. Abortando.")
        sys.exit(1)

    matrix = np.vstack(embeddings).astype("float32")
    print(f"Embeddings generados: {matrix.shape[0]} imágenes × {matrix.shape[1]} dimensiones.")

    indexer.build_index(matrix, kept_paths)

    print(f"Índice construido en '{indexer.BACKEND}' con {matrix.shape[0]} imágenes.")
    print("Inicia el servidor con: uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
