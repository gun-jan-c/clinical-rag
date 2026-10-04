# Decision log

| Date | Decision | Why | Alternatives considered |
| --- | --- | --- | --- |
| 2026-10-03 | Drop Amazon Bedrock from v1; use OpenAI API + Cohere API | New AWS account verification blocked access to every Bedrock model (Error 002) | Wait for AWS verification (blocks the build day) |
| 2026-10-03 | Model IDs: `gpt-6-luna` (generation + fast), `text-embedding-3-small` / `-large` at 1024 dims, Cohere `rerank-v4.0-pro` | All confirmed reachable by `scripts/check_providers.py`. Rerank pro over fast: trial limits count calls, not model tier, so the more accurate model costs nothing extra | `rerank-v4.0-fast` (lower latency), `rerank-v3.5` (older) |
