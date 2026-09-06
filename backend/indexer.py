"""
indexer.py
==========
Capa de búsqueda vectorial con dos backends intercambiables mediante la
variable de entorno ``INDEX_BACKEND``:

- ``pgvector`` (por defecto): los vectores viven en PostgreSQL y la similitud
  coseno se calcula en la base de datos con el operador ``<=>``. Se crea un
  índice HNSW para búsqueda aproximada rápida.
- ``faiss``: índice local ``IndexFlatIP`` en disco. No requiere base de datos
  y sirve para trabajar offline. Como los embeddings están normalizados L2,
  el producto interno equivale a la similitud coseno.

Ambos backends devuelven exactamente la misma estructura de resultados.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import numpy as np

from backend import utils

logger = logging.getLogger(__name__)

BACKEND: str = utils.INDEX_BACKEND

_ready = False

# Estado del modo FAISS.
_faiss_index = None
_faiss_paths: list[str] = []
_faiss_metadata: dict[str, dict] = {}


def _as_float32_1d(vector: np.ndarray) -> np.ndarray:
    """Convierte cualquier arreglo a un vector 1-D ``float32`` contiguo."""
    arr = np.asarray(vector, dtype="float32")
    if arr.ndim > 1:
        arr = arr.flatten()
    return np.ascontiguousarray(arr)


def _meta_for(path: str, metadata: dict[str, dict]) -> dict:
    """Metadatos (categorías y URL pública) de una imagen por su nombre de archivo."""
    return metadata.get(os.path.basename(path), {"categories": [], "url": ""})


# --------------------------------------------------------------------------- #
# Construcción del índice
# --------------------------------------------------------------------------- #

def build_index(embeddings: np.ndarray, image_paths: list[str]) -> None:
    """
    Reemplaza el índice actual con los ``embeddings`` dados.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz ``(N, dim)`` float32 con vectores normalizados L2.
    image_paths : List[str]
        ``N`` rutas relativas a ``utils.IMAGES_DIR``, en el mismo orden.
    """
    n_vectors = int(embeddings.shape[0])
    if n_vectors != len(image_paths):
        raise ValueError(
            f"El número de embeddings ({n_vectors}) no coincide con el de rutas ({len(image_paths)})."
        )
    if n_vectors == 0:
        raise ValueError("No se puede construir un índice con cero embeddings.")

    metadata = utils.load_metadata()

    if BACKEND == "faiss":
        _build_faiss(embeddings, image_paths, metadata)
    else:
        _build_pgvector(embeddings, image_paths, metadata)

    global _ready
    _ready = True


def _build_pgvector(embeddings: np.ndarray, image_paths: list[str], metadata: dict) -> None:
    from psycopg2.extras import Json, execute_batch

    from backend.database import connection, init_db

    init_db()
    logger.info("Insertando %d vectores en PostgreSQL ...", len(image_paths))

    records = []
    for i, path in enumerate(image_paths):
        meta = _meta_for(path, metadata)
        records.append(
            (path, meta["url"] or None, Json(meta["categories"]), _as_float32_1d(embeddings[i]))
        )

    insert_sql = """
        INSERT INTO images (image_path, image_url, categories, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (image_path) DO UPDATE SET
            image_url  = EXCLUDED.image_url,
            categories = EXCLUDED.categories,
            embedding  = EXCLUDED.embedding;
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE images;")
            execute_batch(cur, insert_sql, records, page_size=200)
            # Índice HNSW para búsqueda aproximada (ANN) por distancia coseno.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS images_embedding_hnsw_idx "
                "ON images USING hnsw (embedding vector_cosine_ops);"
            )
    logger.info("Vectores insertados e índice HNSW verificado.")


def _build_faiss(embeddings: np.ndarray, image_paths: list[str], metadata: dict) -> None:
    import faiss

    global _faiss_index, _faiss_paths, _faiss_metadata

    utils.ensure_dirs()
    matrix = np.ascontiguousarray(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, utils.FAISS_INDEX_PATH)
    with open(utils.FAISS_PATHS_PATH, "w", encoding="utf-8") as handle:
        json.dump(list(image_paths), handle, ensure_ascii=False)

    _faiss_index = index
    _faiss_paths = list(image_paths)
    _faiss_metadata = metadata
    logger.info("Índice FAISS guardado en %s (%d vectores).", utils.FAISS_INDEX_PATH, index.ntotal)


# --------------------------------------------------------------------------- #
# Carga al arranque
# --------------------------------------------------------------------------- #

def load_index() -> None:
    """
    Inicializa el backend activo al arrancar el servidor.

    Lanza una excepción si el backend no está disponible (sin BD, sin archivos
    FAISS). ``main.lifespan`` la captura para arrancar en modo degradado.
    """
    global _ready, _faiss_index, _faiss_paths, _faiss_metadata

    if BACKEND == "faiss":
        import faiss

        if not (os.path.exists(utils.FAISS_INDEX_PATH) and os.path.exists(utils.FAISS_PATHS_PATH)):
            raise FileNotFoundError(
                f"No existe el índice FAISS en {utils.EMBEDDINGS_DIR}. "
                "Ejecuta 'python scripts/build_index.py' primero."
            )
        _faiss_index = faiss.read_index(utils.FAISS_INDEX_PATH)
        with open(utils.FAISS_PATHS_PATH, encoding="utf-8") as handle:
            _faiss_paths = json.load(handle)
        _faiss_metadata = utils.load_metadata()
        _ready = True
        logger.info("Índice FAISS cargado: %d vectores.", _faiss_index.ntotal)
        return

    from backend.database import init_db

    init_db()
    _ready = True
    logger.info("PostgreSQL listo: %d vectores en la tabla images.", index_size())


def is_ready() -> bool:
    """``True`` si el índice está cargado y se pueden atender búsquedas."""
    return _ready


def index_size() -> int:
    """Número de vectores indexados (0 si el índice no está listo)."""
    if not _ready:
        return 0
    if BACKEND == "faiss":
        return int(_faiss_index.ntotal) if _faiss_index is not None else 0

    from backend.database import connection

    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM images;")
                row = cur.fetchone()
                return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo contar los vectores: %s", exc)
        return 0


# --------------------------------------------------------------------------- #
# Búsqueda
# --------------------------------------------------------------------------- #

def search(query_embedding: np.ndarray, top_k: int = utils.TOP_K_DEFAULT) -> list[dict]:
    """
    Devuelve las ``top_k`` imágenes más similares al vector de consulta.

    Cada resultado contiene ``image_path``, ``image_url`` (puede ser vacío),
    ``categories`` y ``score`` (similitud coseno, mayor es mejor).
    """
    if not _ready:
        raise RuntimeError("El índice no está listo. Construye el índice y reinicia el servidor.")

    query_vec = _as_float32_1d(query_embedding)
    top_k = max(1, min(int(top_k), utils.TOP_K_MAX))

    if BACKEND == "faiss":
        return _search_faiss(query_vec, top_k)
    return _search_pgvector(query_vec, top_k)


def _search_faiss(query_vec: np.ndarray, top_k: int) -> list[dict]:
    scores, indices = _faiss_index.search(query_vec[np.newaxis, :], top_k)
    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        path = _faiss_paths[idx]
        meta = _meta_for(path, _faiss_metadata)
        results.append(
            {
                "image_path": path,
                "image_url": meta["url"],
                "categories": meta["categories"],
                "score": float(score),
            }
        )
    return results


def _search_pgvector(query_vec: np.ndarray, top_k: int) -> list[dict]:
    from backend.database import connection

    # ``<=>`` es la distancia coseno; 1 - distancia = similitud.
    sql = """
        SELECT image_path, image_url, categories, 1 - (embedding <=> %s) AS similarity
        FROM images
        ORDER BY embedding <=> %s
        LIMIT %s;
    """
    with connection() as conn:
        with conn.cursor() as cur:
            # HNSW devuelve como máximo ef_search candidatos; debe ser >= LIMIT.
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (max(40, top_k * 2),))
            cur.execute(sql, (query_vec, query_vec, top_k))
            rows = cur.fetchall()

    return [
        {
            "image_path": path,
            "image_url": url or "",
            "categories": list(cats) if cats else [],
            "score": float(sim),
        }
        for path, url, cats, sim in rows
    ]


# --------------------------------------------------------------------------- #
# Estadísticas
# --------------------------------------------------------------------------- #

def stats() -> dict:
    """Total de imágenes y conteo por categoría del índice activo."""
    if not _ready:
        return {"backend": BACKEND, "total_images": 0, "categories": []}

    if BACKEND == "faiss":
        counter: Counter = Counter()
        for path in _faiss_paths:
            counter.update(_meta_for(path, _faiss_metadata)["categories"])
        categories = [{"name": name, "count": count} for name, count in counter.most_common()]
        return {"backend": BACKEND, "total_images": len(_faiss_paths), "categories": categories}

    from backend.database import connection

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM images;")
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT category, count(*) AS n
                FROM images, jsonb_array_elements_text(categories) AS category
                WHERE jsonb_typeof(categories) = 'array'
                GROUP BY category
                ORDER BY n DESC, category ASC;
                """
            )
            categories = [{"name": name, "count": int(n)} for name, n in cur.fetchall()]
    return {"backend": BACKEND, "total_images": total, "categories": categories}
