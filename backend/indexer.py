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
    """
    Convierte un arreglo NumPy a un vector 1-D contiguo de tipo ``float32``.

    Garantiza que el arreglo sea unidimensional y esté en memoria contigua
    (C-order), requisito de las librerías FAISS y pgvector al recibir vectores.

    Parameters
    ----------
    vector : np.ndarray
        Arreglo NumPy de cualquier forma y tipo. Si tiene más de una
        dimensión, se aplana automáticamente.

    Returns
    -------
    np.ndarray
        Arreglo 1-D de tipo ``float32`` con disposición de memoria contigua
        (``np.ascontiguousarray``).
    """
    arr = np.asarray(vector, dtype="float32")
    if arr.ndim > 1:
        arr = arr.flatten()
    return np.ascontiguousarray(arr)


def build_index(embeddings: np.ndarray, image_paths: List[str]) -> None:
    """
    Inserta embeddings e imágenes en la tabla ``images`` de PostgreSQL.

    Limpia la tabla existente, luego inserta en lote todos los vectores
    junto con sus rutas de archivo y categorías (obtenidas de ``metadata.json``
    si el archivo existe). Usa ``ON CONFLICT ... DO UPDATE`` para manejar
    duplicados de forma idónea.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz de forma ``(N, embedding_dim)`` con los vectores de embedding
        generados por el modelo CLIP, uno por imagen. Tipo esperado: ``float32``.
    image_paths : List[str]
        Lista de ``N`` rutas de archivo relativas a ``utils.IMAGES_DIR``,
        en el mismo orden que las filas de ``embeddings``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Si el número de embeddings no coincide con el número de rutas,
        o si se pasa una matriz vacía (0 vectores).
    psycopg2.Error
        Cualquier error de PostgreSQL durante la inserción. La transacción
        se revierte (rollback) automáticamente y el error se re-lanza.
    Exception
        Errores de conexión provenientes de ``get_connection()`` o de
        ``init_db()``.

    Notes
    -----
    - Llama a ``init_db()`` internamente para garantizar que la extensión
      pgvector y la tabla ``images`` existan antes de insertar.
    - Usa ``psycopg2.extras.execute_batch`` con ``page_size=100`` para
      mejor rendimiento en inserciones masivas.
    - Al finalizar exitosamente, actualiza la bandera global ``_ready = True``.
    - El archivo ``metadata.json`` debe ubicarse en ``utils.DATA_DIR`` y
      tener la forma ``{"nombre_archivo.jpg": ["categoria1", ...], ...}``.
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
    Inicializa el backend de búsqueda (PostgreSQL o FAISS) al arranque del servidor.

    Si ``USE_FAISS`` es ``True``: lee el índice FAISS desde disco
    (``utils.INDEX_PATH``) y carga los metadatos de rutas e imágenes.
    Si ``USE_FAISS`` es ``False`` (por defecto): invoca ``init_db()`` para
    asegurar el esquema y verifica cuántos vectores hay en PostgreSQL.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Cualquier error de conexión o de I/O al leer archivos FAISS.
        En el flujo normal de FastAPI este error se captura en ``lifespan``
        y el servidor arranca de todas formas en modo degradado (HTTP 503).

    Notes
    -----
    - Para el modo FAISS: si los archivos de índice no existen, establece
      ``_ready = False`` y retorna sin lanzar excepción.
    - Al finalizar exitosamente en cualquier modo, actualiza ``_ready = True``.
    - Esta función debe ser llamada una única vez durante el ciclo de vida
      (lifespan) de la aplicación FastAPI.
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
    """
    Indica si el indexador fue inicializado y está listo para responder búsquedas.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    bool
        ``True`` si ``load_index()`` o ``build_index()`` completaron
        exitosamente; ``False`` en caso contrario.
    """
    return _ready


def index_size() -> int:
    """
    Retorna el número total de vectores almacenados en el backend activo.

    Consulta ``COUNT(*)`` en PostgreSQL (modo por defecto) o lee
    ``_faiss_index.ntotal`` en modo FAISS.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    int
        Número de imágenes/vectores indexados. Retorna ``0`` si el indexador
        no está listo o si ocurre cualquier error durante la consulta.
    """
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
    Recupera las ``top_k`` imágenes más similares al vector de consulta.

    Ejecuta una búsqueda por similitud del coseno contra todos los vectores
    almacenados (PostgreSQL con pgvector o índice FAISS) y retorna los
    resultados ordenados de mayor a menor similitud.

    Parameters
    ----------
    query_embedding : np.ndarray
        Vector de consulta 1-D de tipo ``float32`` con forma
        ``(embedding_dim,)`` (p.ej. ``(512,)`` para ViT-B/32).
        Debe estar normalizado L2 para que la búsqueda sea correcta.
    top_k : int, opcional
        Número máximo de resultados a retornar.
        Por defecto usa ``utils.TOP_K_DEFAULT`` (valor configurado en ``.env``).

    Returns
    -------
    List[Dict]
        Lista de hasta ``top_k`` diccionarios, ordenada de mayor a menor
        similitud. Cada diccionario tiene las siguientes claves:

        - ``"image_path"`` (str): ruta relativa de la imagen en ``IMAGES_DIR``.
        - ``"categories"`` (list): lista de etiquetas/categorías de la imagen
          (puede ser lista vacía si no hay metadatos).
        - ``"score"`` (float): puntuación de similitud en rango ``[0.0, 1.0]``
          (1 - distancia del coseno para PostgreSQL; distancia L2 para FAISS).

    Raises
    ------
    RuntimeError
        Si ``is_ready()`` retorna ``False`` (el indexador no fue inicializado).
    psycopg2.Error
        Cualquier error de PostgreSQL durante la consulta vectorial.
    Exception
        Errores de conexión provenientes de ``get_connection()``.

    Notes
    -----
    - En modo PostgreSQL, el operador ``<=>`` de pgvector calcula la
      **distancia del coseno** (no similitud). La similitud se obtiene
      como ``1 - distancia``, por lo que el rango es ``[-1, 1]``.
    - En modo FAISS, el score es la **distancia L2** (menor es mejor);
      no se invierte para mantener compatibilidad con el índice interno.
    - Esta función es síncrona; en FastAPI debe ejecutarse con
      ``run_in_threadpool`` para no bloquear el event loop.
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
