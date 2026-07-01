import logging
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from backend import utils

logger = logging.getLogger(__name__)

def get_connection():
    """
    Crea y devuelve una conexión activa a la base de datos PostgreSQL de Supabase.

    Usa los parámetros de conexión (host, puerto, nombre de BD, usuario y
    contraseña) definidos en el módulo ``utils``, que a su vez los lee
    desde las variables de entorno o el archivo ``.env``.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    psycopg2.extensions.connection
        Objeto de conexión psycopg2 listo para crear cursores y ejecutar
        sentencias SQL. El llamador es responsable de cerrarlo.

    Raises
    ------
    psycopg2.OperationalError
        Si no es posible establecer conexión con el servidor (host incorrecto,
        credenciales inválidas, puerto bloqueado, etc.). El error se registra
        en el logger antes de propagarse.
    Exception
        Cualquier otro error inesperado durante la conexión.
    """
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
    """
    Inicializa el esquema de la base de datos para el sistema de búsqueda de imágenes.

    Realiza las siguientes operaciones en orden:
    1. Habilita la extensión ``pgvector`` si aún no existe.
    2. Registra el tipo ``vector`` de pgvector con psycopg2.
    3. Crea la tabla ``images`` (si no existe) con columnas para ruta,
       categorías (JSONB) y embedding (VECTOR de 512 dimensiones).

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    None

    Raises
    ------
    psycopg2.Error
        Cualquier error de PostgreSQL durante la creación de la extensión
        o la tabla. La transacción se revierte (rollback) automáticamente
        y el error se registra antes de propagarse.
    Exception
        Errores de conexión provenientes de ``get_connection()``.

    Notes
    -----
    - La tabla ``images`` tiene un índice único implícito en ``image_path``
      (restricción ``UNIQUE NOT NULL``).
    - La dimensión del vector está fija en 512 para el modelo ViT-B/32.
      Si se cambia de modelo, puede ser necesario recrear la tabla.
    - La conexión se cierra en el bloque ``finally`` independientemente del
      resultado.
    """
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
    """
    Ejecuta una sentencia SQL arbitraria y opcionalmente retorna los resultados.

    Abre una conexión nueva para cada llamada, registra pgvector, ejecuta
    la sentencia y, si es un ``SELECT``, retorna las filas. Para cualquier
    otra sentencia (INSERT, UPDATE, DELETE, etc.) realiza un ``commit``.

    Parameters
    ----------
    query : str
        Sentencia SQL a ejecutar. Puede contener marcadores de posición
        (``%s``) compatibles con psycopg2 para parámetros vinculados.
    vars : tuple o list, opcional
        Parámetros a sustituir en los marcadores ``%s`` de ``query``.
        Por defecto es ``None`` (sin parámetros).
    fetch_all : bool, opcional
        Si es ``True`` (por defecto), retorna todas las filas con
        ``cursor.fetchall()`` para consultas SELECT.
        Si es ``False``, retorna solo la primera fila con
        ``cursor.fetchone()``.

    Returns
    -------
    list of tuple o None
        - Para sentencias ``SELECT`` con ``fetch_all=True``: lista de tuplas
          con todas las filas resultantes (puede ser lista vacía).
        - Para sentencias ``SELECT`` con ``fetch_all=False``: una sola tupla
          con la primera fila, o ``None`` si no hay resultados.
        - Para sentencias no-SELECT (INSERT, UPDATE, etc.): ``None``
          (solo se efectúa el commit).

    Raises
    ------
    psycopg2.Error
        Cualquier error de PostgreSQL durante la ejecución. La transacción
        se revierte (rollback) automáticamente y el error se re-lanza.
    Exception
        Errores de conexión provenientes de ``get_connection()``.

    Notes
    -----
    - Esta función es adecuada para consultas puntuales. Para operaciones
      masivas (batch inserts) se recomienda usar ``psycopg2.extras.execute_batch``
      directamente con una conexión dedicada (como en ``indexer.build_index``).
    - La conexión se cierra en el bloque ``finally`` independientemente del
      resultado.
    """
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
