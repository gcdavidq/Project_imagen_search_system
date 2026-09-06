"""
search_cli.py
=============
Búsqueda interactiva por texto desde la terminal, sin levantar el servidor.
Usa exactamente los mismos módulos que la API, así que sirve para validar
el índice (pgvector o FAISS) y el modelo antes de desplegar.

    python scripts/search_cli.py
    python scripts/search_cli.py --backend faiss --top-k 10
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Búsqueda texto → imagen interactiva.")
    parser.add_argument("--backend", choices=["pgvector", "faiss"], default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.backend:
        os.environ["INDEX_BACKEND"] = args.backend

    from backend import embedder, indexer, utils

    print(f"Backend: {indexer.BACKEND} | Modelo: {utils.MODEL_NAME}/{utils.PRETRAINED}")
    print("Cargando modelo e índice ...")
    embedder.load_model(utils.MODEL_NAME, utils.PRETRAINED)
    indexer.load_index()
    print(f"Índice listo: {indexer.index_size()} imágenes. Escribe 'salir' para terminar.\n")

    while True:
        try:
            query = input("Consulta > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"salir", "exit", "quit"}:
            break
        if not query:
            continue

        vector = embedder.get_text_embedding(query)
        hits = indexer.search(vector, top_k=args.top_k)
        for i, hit in enumerate(hits, start=1):
            cats = ", ".join(hit["categories"]) or "sin categorías"
            print(f"{i:>2}. {hit['score']:.4f}  {hit['image_path']}  [{cats}]")
        print()


if __name__ == "__main__":
    main()
