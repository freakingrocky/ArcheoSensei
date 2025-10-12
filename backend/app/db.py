import psycopg
from contextlib import contextmanager
from .settings import settings
from pgvector.psycopg import register_vector

def get_conn():
    conn = psycopg.connect(settings.DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn

@contextmanager
def conn_cursor():
    with get_conn() as conn:
        with conn.cursor() as cur:
            yield cur
