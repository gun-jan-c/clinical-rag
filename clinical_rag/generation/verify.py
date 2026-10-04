"""Automatic checks on every claim (ProjectSpec.md section 5.7). Failing claims are flagged, never dropped.

- Has citation: the claim cites at least one chunk.
- Citation valid: every cited chunk was among the sources the model was given.
- Numbers grounded: every number in the claim appears in the text of at least one cited source, as given to
  the model (for a long section that is the matched piece, not the whole section). Whole numbers only, so
  "4.9" is not found inside "14.9". Thousands commas are ignored ("1,961" = "1961"), and so are signs
  ("−14.9" in a table matches "14.9% reduction" in a claim).
"""

import re

from clinical_rag.schemas import Claim

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def numbers(text: str) -> list[str]:
    """Distinct numbers in a text, in order: '15.8% vs 6.1% at week 72' -> ['15.8', '6.1', '72']."""
    return list(dict.fromkeys(NUMBER_RE.findall(THOUSANDS_COMMA_RE.sub("", text))))


def appears(number: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", THOUSANDS_COMMA_RE.sub("", text)) is not None


def verify(claims: list[Claim], sources: dict[str, str]) -> list[Claim]:
    """sources: chunk_id -> the text the model was given for it. Returns the claims with verified/issues set."""
    out = []
    for claim in claims:
        issues = []
        if not claim.citation_ids:
            issues.append("no citation")
        invalid = [i for i in claim.citation_ids if i not in sources]
        if invalid:
            issues.append(f"cites chunks that were not among the sources: {', '.join(invalid)}")
        cited = [sources[i] for i in claim.citation_ids if i in sources]
        missing = [n for n in numbers(claim.text) if not any(appears(n, t) for t in cited)]
        if missing:
            issues.append(f"numbers not found in the cited sources: {', '.join(missing)}")
        out.append(claim.model_copy(update={"verified": not issues, "issues": issues}))
    return out
