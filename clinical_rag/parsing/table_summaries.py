"""One-line table summaries by the fast model, added to `section` table chunks (ProjectSpec.md section 5.5).

A table chunk's `content` (what is embedded and searched) becomes: title/section prefix, label + caption,
the summary, the header row. The full table stays in `context`, which is what the LLM reads. The summary is
asked not to contain numbers, so a wrong number can never enter the brief through it.

Run from the repo root:  python -m clinical_rag.parsing.table_summaries [--dry-run]
Only table chunks without `metadata.summary` are summarized, so a re-run continues where it stopped.
"""

import os
import sys

from psycopg.types.json import Jsonb

from clinical_rag.chunking.strategies import count_tokens
from clinical_rag.config import cost_usd
from clinical_rag.db import connect
from clinical_rag.llm import get_chat

PROMPT = """Below is a table from a clinical document.

{table}

Write ONE sentence (at most 30 words) saying what this table reports: the measures, the groups or doses \
compared, and the time point if shown. Do not include any numbers or results. Reply with the sentence only."""

BATCH = 50  # chunks per database commit
CONCURRENCY = 8  # parallel model calls


def add_summary(content: str, summary: str) -> str:
    """Put the summary on its own line just before the header row (the last line of a table chunk)."""
    before, header = content.rsplit("\n", 1)
    return f"{before}\n{summary}\n{header}"


def pending(conn) -> list[tuple[str, str, str]]:
    return conn.execute(
        "select chunk_id, content, context from chunks where strategy = 'section' "
        "and metadata->>'content_type' = 'table' and not metadata ? 'summary' order by chunk_id").fetchall()


def main() -> None:
    model_id = os.environ["FAST_MODEL_ID"]
    with connect() as conn:
        rows = pending(conn)
        prompt_tokens = sum(count_tokens(PROMPT.format(table=r[2])) for r in rows)
        print(f"{len(rows)} tables to summarize, ~{prompt_tokens:,} input tokens, "
              f"~${cost_usd(model_id, prompt_tokens, 200 * len(rows)):.2f} with {model_id}")  # ~200 out incl. reasoning
        if "--dry-run" in sys.argv or not rows:
            return
        chat = get_chat(fast=True, max_retries=6)
        tokens_in = tokens_out = 0
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            replies = chat.batch([PROMPT.format(table=r[2]) for r in batch], config={"max_concurrency": CONCURRENCY})
            for (chunk_id, content, _), reply in zip(batch, replies):
                summary = " ".join(reply.content.split())
                new_content = add_summary(content, summary)
                conn.execute(
                    "update chunks set content = %s, token_count = %s, metadata = metadata || %s "
                    "where chunk_id = %s",
                    (new_content, count_tokens(new_content), Jsonb({"summary": summary}), chunk_id))
                # The content changed, so any vector made from the old content is stale.
                conn.execute("delete from chunk_embeddings where chunk_id = %s", (chunk_id,))
                usage = reply.usage_metadata or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
            conn.commit()
            print(f"  {start + len(batch)}/{len(rows)}  spent ${cost_usd(model_id, tokens_in, tokens_out):.3f}")
        print(f"done: {tokens_in:,} input + {tokens_out:,} output tokens = ${cost_usd(model_id, tokens_in, tokens_out):.3f}")


if __name__ == "__main__":
    main()
