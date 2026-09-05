"""
database.py
===========
Acceso a PostgreSQL + pgvector mediante un pool de conexiones psycopg2.

Solo se utiliza cuando ``INDEX_BACKEND=pgvector``. Soporta configurar la
conexión con ``DATABASE_URL`` (cadena completa, como la que entregan Neon o Supabase)
o con las variables sueltas ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``
y ``DB_PASSWORD``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2 import pool as pg_pool

from backend import utils

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _connection_kwargs() -> dict:
    """Construye los argumentos de conexión a partir de la configuración."""
    if utils.DATABASE_URL:
        return {"dsn": utils.DATABASE_URL}
    if not utils.DB_HOST:
        raise RuntimeError(
            "Base de datos no configurada: define DATABASE_URL o DB_HOST/DB_PASSWORD en .env "
            "(o usa INDEX_BACKEND=faiss para trabajar sin PostgreSQL)."
        )
    return {
        "host": utils.DB_HOST,
        "port": utils.DB_PORT,
        "dbname": utils.DB_NAME,
        "user": utils.DB_USER,
        "password": utils.DB_PASSWORD,
    }


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Devuelve el pool de conexiones, creándolo perezosamente la primera vez."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            kwargs = _connection_kwargs()
            logger.info("Creando pool de conexiones PostgreSQL ...")
            try:
                _pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **kwargs)
            except psycopg2.Error as exc:
                logger.error("No se pudo conectar a PostgreSQL: %s", exc)
                raise
    return _pool


def close_pool() -> None:
    """Cierra todas las conexiones del pool (llamar al apagar el servidor)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def connection(register: bool = True) -> Iterator[psycopg2.extensions.connection]:
    """
    Toma una conexión del pool y la devuelve al salir del bloque.

    Hace ``commit`` si el bloque termina bien y ``rollback`` si lanza una
    excepción. Con ``register=True`` registra el tipo ``vector`` de pgvector
    para poder pasar arreglos NumPy directamente como parámetros.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        if register:
            register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db() -> None:
    """
    Crea la extensión ``vector`` y la tabla ``images`` si no existen.

    Es idempotente: se puede ejecutar en cada arranque. También agrega la
    columna ``image_url`` a tablas creadas por versiones anteriores.
    """
    with connection(register=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS images (
                    id          SERIAL PRIMARY KEY,
                    image_path  TEXT UNIQUE NOT NULL,
                    image_url   TEXT,
                    categories  JSONB,
                    embedding   VECTOR({utils.EMBEDDING_DIM})
                );
                """
            )
            cur.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS image_url TEXT;")
    logger.info("Esquema de base de datos verificado.")
