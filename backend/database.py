import logging
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from backend import utils

logger = logging.getLogger(__name__)

def get_connection():
    """Devuelve una conexión psycopg2 a la base de datos de Supabase."""
    try:
        conn = psycopg2.connect(
            host=utils.DB_HOST,
            port=utils.DB_PORT,
            dbname=utils.DB_NAME,
            user=utils.DB_USER,
            password=utils.DB_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Error al conectar con la base de datos: {e}")
        raise

def init_db():
    """Inicializa el esquema de la base de datos para el sistema de búsqueda de imágenes."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Habilitar la extensión pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Registrar el tipo vector con psycopg2
            register_vector(conn)
            
            # Crear la tabla images
            cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT UNIQUE NOT NULL,
                    categories JSONB,
                    embedding VECTOR(512)
                );
            """)
            

            
        conn.commit()
        logger.info("Base de datos inicializada exitosamente.")
    except Exception as e:
        logger.error(f"Error al inicializar la base de datos: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def execute_query(query: str, vars=None, fetch_all=True):
    conn = get_connection()
    try:
        # Registrar pgvector antes de realizar la consulta
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(query, vars)
            if query.strip().upper().startswith("SELECT"):
                if fetch_all:
                    return cur.fetchall()
                return cur.fetchone()
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
