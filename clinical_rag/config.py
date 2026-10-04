"""Settings that are not secrets.

PRICES: US dollars per 1M tokens, from https://developers.openai.com/api/docs/pricing (Standard tier),
checked 2026-10-04. Embedding models have no output price.
"""

PRICES = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
}


def cost_usd(model_id: str, input_tokens: int, output_tokens: int = 0) -> float:
    price = PRICES[model_id]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
