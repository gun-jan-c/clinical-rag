"""Apply SQL files in db/migrations/ to DATABASE_URL, in name order, skipping ones already applied.

Run from the repo root:  python -m scripts.apply_migrations
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"


def main() -> None:
    load_dotenv()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("create table if not exists schema_migrations "
                     "(name text primary key, applied_at timestamptz not null default now())")
        conn.execute("alter table schema_migrations enable row level security")  # same lock as every other table
        applied = {row[0] for row in conn.execute("select name from schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                print(f"skip   {path.name}")
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            conn.commit()
            print(f"applied {path.name}")


if __name__ == "__main__":
    main()
