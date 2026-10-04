"""HTML / JATS tables -> markdown, keeping header rows and footnotes (ProjectSpec.md section 5.5)."""

from lxml import etree


def clean(text: str) -> str:
    return " ".join(text.split())


def _local(tag) -> str:
    """Tag name without namespace; '' for comments and processing instructions."""
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _cells(row: etree._Element) -> list[str]:
    cells = []
    for cell in row:
        if _local(cell.tag) not in ("td", "th"):
            continue
        for sup in cell.iter():
            if _local(sup.tag) == "sup" and sup.text:
                sup.text = "^" + sup.text  # footnote marker: 'Diarrhea^a', not 'Diarrheaa'
        text = clean("".join(cell.itertext())).replace("|", "\\|")
        cells += [text] + [""] * (int(cell.get("colspan", "1") or 1) - 1)  # spanned columns stay empty
    return cells


def table_to_markdown(table: etree._Element) -> tuple[str, str]:
    """Returns (markdown, footnotes). Rows inside <tfoot> are footnotes, not data."""
    rows, foot = [], []
    for row in table.iter():
        if _local(row.tag) != "tr":
            continue
        in_foot = any(_local(a.tag) == "tfoot" for a in row.iterancestors())
        if in_foot:
            foot.append(clean(" ".join(row.itertext())))
        elif cells := _cells(row):
            rows.append(cells)
    if not rows:
        return "", "\n".join(foot)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines), "\n".join(f for f in foot if f)
