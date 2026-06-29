"""
indexer.py

"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Dict, Optional

import numpy as np

from backend import utils
from backend.database import get_connection, init_db, execute_query
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

# Bandera de estado
_ready = False

# Toggle para usar FAISS o PostgreSQL
USE_FAISS = False

# Variables para FAISS
_faiss_index = None
_faiss_paths = []
_faiss_metadata = {}


def _as_float32_1d(vector: np.ndarray) -> np.ndarray:
    """Devolver un arreglo unidimensional ``float32`` contiguo."""
    arr = np.asarray(vector, dtype="float32")
    if arr.ndim > 1:
        arr = arr.flatten()
    return np.ascontiguousarray(arr)


def build_index(embeddings: np.ndarray, image_paths: List[str]) -> None:
    """
    Insertar incrustaciones (embeddings) y sus rutas en la tabla `images` de PostgreSQL.
    También inyecta categorías de `metadata.json` si están disponibles.
    """
    # Asegurarnos de que la base de datos (y la extensión vector) estén inicializadas
    init_db()
    
    n_vectors = embeddings.shape[0]

    if n_vectors != len(image_paths):
        raise ValueError(
            f"El número de incrustaciones ({n_vectors}) no coincide con el número de "
            f"rutas de imagen ({len(image_paths)})."
        )
    if n_vectors == 0:
        raise ValueError("No se puede construir un índice con cero incrustaciones.")

    logger.info("Insertando %d vectores en PostgreSQL ...", n_vectors)

    # Cargar metadatos si existen
    metadata_path = os.path.join(utils.DATA_DIR, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    conn = get_connection()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # Limpiar datos existentes para un índice nuevo (opcional, pero consistente con el comportamiento de FAISS)
            cur.execute("TRUNCATE TABLE images;")
            
            # Usar execute_batch para un mejor rendimiento
            from psycopg2.extras import execute_batch
            
            insert_query = """
                INSERT INTO images (image_path, categories, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (image_path) DO UPDATE SET 
                    categories = EXCLUDED.categories,
                    embedding = EXCLUDED.embedding;
            """
            
            records = []
            for i in range(n_vectors):
                path = image_paths[i]
                filename = os.path.basename(path)
                cats = metadata.get(filename, [])
                vec = _as_float32_1d(embeddings[i])
                records.append((path, Json(cats), vec))
                
            execute_batch(cur, insert_query, records, page_size=100)
            
        conn.commit()
        logger.info("Vectores insertados exitosamente.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error al insertar vectores: {e}")
        raise
    finally:
        conn.close()

    global _ready
    _ready = True


def load_index() -> None:
    """
    Inicializa la conexión y el esquema de Postgres, o carga el índice FAISS.
    """
    global _ready, _faiss_index, _faiss_paths, _faiss_metadata
    
    if USE_FAISS:
        import faiss
        logger.info("Inicializando el indexador FAISS ...")
        if not os.path.exists(utils.INDEX_PATH) or not os.path.exists(utils.PATHS_PATH):
            logger.warning("No se encontraron los archivos del índice FAISS.")
            _ready = False
            return
            
        _faiss_index = faiss.read_index(utils.INDEX_PATH)
        with open(utils.PATHS_PATH, "r", encoding="utf-8") as f:
            _faiss_paths = json.load(f)
            
        metadata_path = os.path.join(utils.DATA_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                _faiss_metadata = json.load(f)
                
        logger.info("Índice FAISS cargado. Vectores totales: %d", _faiss_index.ntotal)
        _ready = True
    else:
        logger.info("Inicializando el indexador PostgreSQL ...")
        init_db()
        
        # Comprobar si hay datos
        size = index_size()
        logger.info("PostgreSQL listo. Vectores actuales en la base de datos: %d", size)
        _ready = True


def is_ready() -> bool:
    """Devolver ``True`` si la conexión a la base de datos fue validada."""
    return _ready


def index_size() -> int:
    """Devolver el número de vectores actualmente en la base de datos o en el índice FAISS."""
    if not is_ready():
        return 0
    if USE_FAISS:
        return _faiss_index.ntotal if _faiss_index else 0
        
    try:
        res = execute_query("SELECT count(*) FROM images;")
        return res[0][0] if res else 0
    except Exception:
        return 0


def search(query_embedding: np.ndarray, top_k: int = utils.TOP_K_DEFAULT) -> List[Dict]:
    """
    Devolver las ``top_k`` imágenes más similares a ``query_embedding`` desde Postgres o FAISS.
    """
    if not is_ready():
        raise RuntimeError("El indexador no está listo. Llame a load_index() primero.")

    query_vec = _as_float32_1d(query_embedding)
    
    if USE_FAISS:
        import faiss
        query_vec_2d = np.expand_dims(query_vec, axis=0)
        distances, indices = _faiss_index.search(query_vec_2d, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            path = _faiss_paths[idx]
            filename = os.path.basename(path)
            cats = _faiss_metadata.get(filename, [])
            results.append({
                "image_path": path,
                "categories": cats,
                "score": float(dist)
            })
        return results
    
    # <=> es la distancia del coseno. 1 - distancia = similitud. 
    sql = """
        SELECT image_path, categories, 1 - (embedding <=> %s) AS similarity
        FROM images
        ORDER BY embedding <=> %s
        LIMIT %s;
    """
    
    conn = get_connection()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, (query_vec, query_vec, top_k))
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                path, cats, sim = row
                # psycopg2 maneja el análisis sintáctico de jsonb de forma nativa al usar Json
                results.append({
                    "image_path": path,
                    "categories": cats if cats else [],
                    "score": float(sim)
                })
            return results
    finally:
        conn.close()
