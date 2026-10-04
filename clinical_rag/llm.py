"""Model client factories. The only module that names a model provider.

Swapping providers (e.g. to Bedrock in Phase 2) means changing this file and .env, nothing else.
"""

import os

from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Our short names (stored in the database) -> provider model IDs.
EMBEDDING_MODELS = {
    "oai-3-small": "text-embedding-3-small",
    "oai-3-large": "text-embedding-3-large",
}
EMBEDDING_DIMENSIONS = 1024  # both models shortened to the same size so they share one vector column


def get_chat(fast: bool = False, **kwargs) -> ChatOpenAI:
    """Generation model, or the fast model for small jobs (query expansion, table summaries)."""
    model = os.environ["FAST_MODEL_ID" if fast else "GEN_MODEL_ID"]
    return ChatOpenAI(model=model, **kwargs)


def get_embeddings(name: str = "oai-3-small") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODELS[name], dimensions=EMBEDDING_DIMENSIONS)


def get_reranker(top_n: int = 8) -> CohereRerank:
    return CohereRerank(model=os.environ["RERANK_MODEL_ID"], top_n=top_n)
