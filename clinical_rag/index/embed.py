"""Embed chunk `content` with one or both OpenAI models -> `chunk_embeddings` (ProjectSpec.md section 5.6).

Run from the repo root:  python -m clinical_rag.index.embed [--model oai-3-small|oai-3-large] [--dry-run]
Without --model, both models run. Only chunks that have no vector for that model are embedded, so a re-run
with no changes embeds 0 chunks, and a stopped run continues where it left off. `section` table chunks wait
until their summary exists (parsing/table_summaries.py), because the summary changes what gets embedded.
"""

import sys

import numpy as np

from clinical_rag.config import cost_usd
from clinical_rag.db import connect
from clinical_rag.llm import EMBEDDING_MODELS, get_embeddings

MAX_BATCH_TOKENS = 100_000  # OpenAI allows 300k tokens per request; stay well under
MAX_BATCH_ITEMS = 1_000


def batches(rows: list[tuple[str, str, int]]) -> list[list[tuple[str, str, int]]]:
    """Group (chunk_id, content, token_count) rows so each group fits in one request."""
    out, current, tokens = [], [], 0
    for row in rows:
        if current and (tokens + row[2] > MAX_BATCH_TOKENS or len(current) == MAX_BATCH_ITEMS):
            out.append(current)
            current, tokens = [], 0
        current.append(row)
        tokens += row[2]
    if current:
        out.append(current)
    return out


def pending(conn, model: str) -> list[tuple[str, str, int]]:
    return conn.execute(
        "select c.chunk_id, c.content, c.token_count from chunks c "
        "where not exists (select 1 from chunk_embeddings e where e.chunk_id = c.chunk_id and e.model = %s) "
        "and not (c.strategy = 'section' and c.metadata->>'content_type' = 'table' and not c.metadata ? 'summary') "
        "order by c.chunk_id", (model,)).fetchall()


def embed_model(conn, model: str, dry_run: bool) -> None:
    model_id = EMBEDDING_MODELS[model]
    rows = pending(conn, model)
    total = sum(r[2] for r in rows)
    print(f"{model}: {len(rows)} chunks to embed, {total:,} tokens, ~${cost_usd(model_id, total):.2f}")
    if dry_run or not rows:
        return
    embedder = get_embeddings(model, max_retries=6, chunk_size=MAX_BATCH_ITEMS)
    done = spent = 0
    for batch in batches(rows):
        vectors = embedder.embed_documents([r[1] for r in batch])
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunk_embeddings (chunk_id, model, embedding) values (%s, %s, %s) "
                "on conflict (chunk_id, model) do update set embedding = excluded.embedding",
                [(r[0], model, np.array(v, dtype=np.float32)) for r, v in zip(batch, vectors)])
        conn.commit()
        done += len(batch)
        spent += sum(r[2] for r in batch)
        print(f"  {done}/{len(rows)}  ~${cost_usd(model_id, spent):.3f}")


def main() -> None:
    models = [sys.argv[sys.argv.index("--model") + 1]] if "--model" in sys.argv else list(EMBEDDING_MODELS)
    with connect() as conn:
        for model in models:
            embed_model(conn, model, "--dry-run" in sys.argv)


if __name__ == "__main__":
    main()
