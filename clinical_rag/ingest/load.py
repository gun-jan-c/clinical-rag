"""Upsert ingested rows. Documents whose content_hash changed lose their old chunks so they get re-chunked."""

import hashlib
import json

import psycopg
from psycopg.types.json import Jsonb

TRIAL_COLUMNS = ["nct_id", "title", "phase", "status", "sponsor", "enrollment", "first_posted", "start_date",
                 "completion_date", "interventions", "conditions", "has_results", "last_updated"]


def content_hash(*parts) -> str:
    """Fingerprint of a document's content (title, sections, tables, ...); changes when any part changes."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def upsert_trials(conn: psycopg.Connection, trials: list[dict]) -> None:
    cols = ", ".join(TRIAL_COLUMNS)
    placeholders = ", ".join(f"%({c})s" for c in TRIAL_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in TRIAL_COLUMNS[1:])
    with conn.cursor() as cur:
        cur.executemany(f"insert into trials ({cols}) values ({placeholders}) "
                        f"on conflict (nct_id) do update set {updates}", trials)


def upsert_documents(conn: psycopg.Connection, docs: list[dict]) -> dict[str, int]:
    """Returns counts of new, changed, and unchanged documents."""
    existing = dict(conn.execute("select doc_id, content_hash from documents "
                                 "where doc_id = any(%s)", ([d["doc_id"] for d in docs],)).fetchall())
    counts = {"new": 0, "changed": 0, "unchanged": 0}
    for d in docs:
        old_hash = existing.get(d["doc_id"])
        if old_hash is None:
            counts["new"] += 1
        elif old_hash != d["content_hash"]:
            counts["changed"] += 1
            conn.execute("delete from chunks where doc_id = %s", (d["doc_id"],))
        else:
            counts["unchanged"] += 1
        # Metadata (e.g. trial status) is refreshed even when the text is unchanged.
        conn.execute(
            "insert into documents (doc_id, source, url, title, published_date, license, sections, metadata, "
            "content_hash) values (%(doc_id)s, %(source)s, %(url)s, %(title)s, %(published_date)s, %(license)s, "
            "%(sections)s, %(metadata)s, %(content_hash)s) "
            "on conflict (doc_id) do update set url = excluded.url, title = excluded.title, "
            "published_date = excluded.published_date, license = excluded.license, sections = excluded.sections, "
            "metadata = excluded.metadata, content_hash = excluded.content_hash, ingested_at = now()",
            {**d, "sections": Jsonb(d["sections"]), "metadata": Jsonb(d["metadata"])})
    return counts
