"""The in-scope drugs (ProjectSpec.md section 2) and tagging text with them."""

DRUGS = ["semaglutide", "tirzepatide", "liraglutide", "orforglipron",
         "retatrutide", "survodutide", "cagrilintide", "mazdutide"]


def matched_drugs(text: str, aliases: dict[str, str]) -> list[str]:
    """In-scope generic names mentioned in text, by generic name or any alias (code or brand name)."""
    text = text.lower()
    names = {d: d for d in DRUGS} | aliases
    return sorted({generic for name, generic in names.items() if name in text})


def search_terms(aliases: dict[str, str]) -> list[str]:
    """Every name to search the source APIs for: generic names plus all aliases."""
    return DRUGS + sorted(aliases)
