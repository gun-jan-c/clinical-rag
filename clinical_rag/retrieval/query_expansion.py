"""Query expansion: add the other names of every drug the query mentions (ProjectSpec.md section 3, step 2).

"LY3502970 weight loss" also searches "orforglipron"; "Wegovy nausea" also searches "semaglutide", "ozempic", ...
Only the keyword side of hybrid search uses the expanded text; the embedding is made from the original query,
because meaning search already links a brand name to its drug.
"""

from clinical_rag.ingest.drugs import matched_drugs


def expand(query: str, aliases: dict[str, str]) -> str:
    """The query plus every other name of each drug it mentions. Multi-word names are quoted as phrases."""
    lowered = query.lower()
    extra = []
    for drug in matched_drugs(query, aliases):
        for name in [drug, *sorted(a for a, g in aliases.items() if g == drug)]:
            if name not in lowered:
                extra.append(f'"{name}"' if " " in name else name)
    return " ".join([query, *extra])
