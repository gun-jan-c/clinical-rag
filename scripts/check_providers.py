"""Pre-flight smoke test: can we reach every model the app needs?

Run from the repo root:  python -m scripts.check_providers
"""

import sys

from langchain_core.documents import Document
from pydantic import BaseModel

from clinical_rag.llm import EMBEDDING_DIMENSIONS, EMBEDDING_MODELS, get_chat, get_embeddings, get_reranker


class Ping(BaseModel):
    answer: str


def check_chat():
    for fast in (False, True):
        msg = get_chat(fast=fast).invoke("Reply with the single word: ok")
        usage = msg.usage_metadata or {}
        print(f"  {msg.response_metadata.get('model_name')}: {msg.content!r} "
              f"({usage.get('input_tokens')} in / {usage.get('output_tokens')} out tokens)")


def check_structured_output():
    result = get_chat().with_structured_output(Ping).invoke("Set answer to 'ok'.")
    assert isinstance(result, Ping), f"expected Ping, got {type(result)}"
    print(f"  returned {result!r}")


def check_embeddings():
    for name in EMBEDDING_MODELS:
        vector = get_embeddings(name).embed_query("tirzepatide weight loss")
        assert len(vector) == EMBEDDING_DIMENSIONS, f"{name}: got {len(vector)} dimensions"
        print(f"  {name}: {len(vector)} dimensions")


def check_rerank():
    docs = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Tirzepatide reduced body weight in adults with obesity."),
    ]
    ranked = get_reranker(top_n=2).compress_documents(docs, "obesity drug weight loss")
    top = ranked[0]
    assert "Tirzepatide" in top.page_content, "reranker put the wrong document first"
    print(f"  top hit: {top.page_content!r} (score {top.metadata['relevance_score']:.3f})")


def main() -> int:
    checks = [check_chat, check_structured_output, check_embeddings, check_rerank]
    failed = 0
    for check in checks:
        print(f"{check.__name__} ...")
        try:
            check()
            print("  PASS")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
