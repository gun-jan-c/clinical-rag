"""Database connection. psycopg + pgvector (not supabase-py), so moving off Supabase is a connection-string change."""

import os
from functools import cache

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()  # the pool requires new connections to be idle


@cache
def pool() -> ConnectionPool:
    """Open connections kept for reuse by the app. A new connection to Railway takes ~2 s; a query ~0.2 s."""
    return ConnectionPool(os.environ["DATABASE_URL"], min_size=1, max_size=4, configure=_configure, open=True)


def drug_aliases(conn: psycopg.Connection) -> dict[str, str]:
    """Lowercase alias -> generic name, e.g. 'ly3502970' -> 'orforglipron'."""
    return dict(conn.execute("select alias, generic from drug_synonyms").fetchall())
