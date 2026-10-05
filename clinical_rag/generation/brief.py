"""A full brief (ProjectSpec.md sections 2 and 3). Saving it, the daily cap and the audit log are in services/api.py.

Order: pipeline (SQL, instant) -> the LLM sections at the same time -> evidence_gaps last, because it uses the
other sections' not_found items. A section that crashes is marked "failed" and the rest of the brief still finishes.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

from clinical_rag.db import pool
from clinical_rag.generation.cost import total
from clinical_rag.generation.sections import (
    Progress,
    evidence_gaps_section,
    llm_section,
    pipeline_section,
)
from clinical_rag.generation.templates import SECTION_TITLES, TEMPLATES
from clinical_rag.schemas import Brief, BriefRequest, SectionDraft, SectionKey

log = logging.getLogger(__name__)


def _guarded(key: SectionKey, progress: Progress, section, *args) -> SectionDraft:
    """Run one section; any error marks just that section failed (the full error goes to the log)."""
    try:
        return section(*args)
    except Exception as e:
        log.exception("Section %s failed", key)
        progress(key, "failed")
        return SectionDraft(section_key=key, title=SECTION_TITLES[key], review_status="failed",
                            not_found=[f"Section could not be generated ({type(e).__name__})"])


def data_as_of() -> date:
    """Date of the last ingestion run."""
    with pool().connection() as conn:
        return conn.execute("select max(ingested_at)::date from documents").fetchone()[0]


def build_brief(req: BriefRequest, on_progress: Progress | None = None) -> Brief:
    start = time.perf_counter()
    progress = on_progress or (lambda key, step: None)
    done: dict[str, SectionDraft] = {}
    if "pipeline" in req.sections:
        done["pipeline"] = _guarded("pipeline", progress, pipeline_section, req.drugs, progress)
    llm_keys = [k for k in req.sections if k in TEMPLATES]
    with ThreadPoolExecutor(max_workers=max(1, len(llm_keys))) as pool_:
        futures = {k: pool_.submit(_guarded, k, progress, llm_section, k, req.drugs, progress) for k in llm_keys}
        done |= {k: f.result() for k, f in futures.items()}
    if "evidence_gaps" in req.sections:
        open_questions = [q for s in done.values() if s.review_status != "failed" for q in s.not_found]
        done["evidence_gaps"] = _guarded("evidence_gaps", progress, evidence_gaps_section, req.drugs, open_questions,
                                         progress)

    sections = [done[k] for k in req.sections]
    llm = [s for s in sections if s.section_key != "pipeline"]
    status = "failed" if llm and all(s.review_status == "failed" for s in llm) else "draft"
    usage = total([s.usage for s in sections], round((time.perf_counter() - start) * 1000))
    return Brief(brief_id=str(uuid.uuid4()), request=req, data_as_of=data_as_of(), status=status, sections=sections,
                 usage=usage, created_at=datetime.now(timezone.utc))
