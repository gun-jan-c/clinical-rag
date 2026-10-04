"""Database connection. psycopg + pgvector (not supabase-py), so moving off Supabase is a connection-string change."""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def drug_aliases(conn: psycopg.Connection) -> dict[str, str]:
    """Lowercase alias -> generic name, e.g. 'ly3502970' -> 'orforglipron'."""
    return dict(conn.execute("select alias, generic from drug_synonyms").fetchall())
