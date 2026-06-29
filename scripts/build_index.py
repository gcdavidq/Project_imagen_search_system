
from __future__ import annotations

import argparse
import os
import sys

# Make the ``backend`` package importable when this script is run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  
from PIL import Image 
from tqdm import tqdm  

from backend import embedder, indexer, utils  


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pgvector index from images in data/images/.")
    parser.add_argument(
        "--images-dir",
        default=utils.IMAGES_DIR,
        help="Directory containing the dataset images.",
    )
    args = parser.parse_args()

    images_dir = args.images_dir
    # OBTENER LA RUTA DE TODAS LAS IMAGENES DE LA CARPETA DATA/IMAGES/
    image_paths = utils.list_image_files(images_dir)

    if not image_paths:
        print(f"[ERROR] No images found in '{images_dir}'.")
        print("Run 'python scripts/download_dataset.py' first.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {images_dir}.")
    print("Loading CLIP model ...")
    # PASO 2: Carga el modelo CLIP en la memoria 
    embedder.load_model(utils.MODEL_NAME, utils.PRETRAINED)

    embeddings: list[np.ndarray] = []
    kept_paths: list[str] = []

    # Bucle principal. Recorre una por una todas las imágenes encontradas.
    for rel_path in tqdm(image_paths, desc="Embedding images", unit="img"):
        abs_path = os.path.join(images_dir, rel_path)
        try:
            with Image.open(abs_path) as img:
                image = img.convert("RGB")
            # CLIP procesa la imagen y devuelve su embedding
            # (un vector matemático numérico, por ejemplo, de 512 dimensiones).
            vector = embedder.get_image_embedding(image)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  skipped {rel_path}: {exc}")
            continue

        embeddings.append(vector)
        kept_paths.append(rel_path)

    if not embeddings:
        print("[ERROR] No images could be embedded. Aborting.")
        sys.exit(1)

    # PASO 4: CREACIÓN DE LA MATRIZ. 
    # Apila (junta) todos los vectores individuales en una sola gran matriz bidimensional.
    # La librería psycopg2 (junto con pgvector) requiere este formato para enviarlo a PostgreSQL.
    matrix = np.vstack(embeddings).astype("float32")
    print(f"Embedded {matrix.shape[0]} images -> vectors of dimension {matrix.shape[1]}.")

    # PASO 5: CONSTRUCCIÓN Y GUARDADO DEL ÍNDICE.
    # Inserta la matriz numérica en la base de datos PostgreSQL alojada en Supabase,
    # junto con las rutas de las imágenes y las categorías cargadas del metadata.json.
    indexer.build_index(matrix, kept_paths)

    print(f" Index built and uploaded to PostgreSQL with {matrix.shape[0]} images")
    print("Start the server with: uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
