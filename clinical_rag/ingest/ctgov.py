"""ClinicalTrials.gov API v2: obesity trials for the in-scope drugs -> `trials` rows + `documents` rows.

Run from the repo root:  python -m clinical_rag.ingest.ctgov
"""

from datetime import date

import requests

from clinical_rag.db import connect, drug_aliases
from clinical_rag.ingest.drugs import matched_drugs, search_terms
from clinical_rag.ingest.load import content_hash, upsert_documents, upsert_trials

API_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_studies(search_terms: list[str]) -> list[dict]:
    """All obesity studies mentioning any search term as an intervention (follows every page)."""
    params = {"query.cond": "obesity", "query.intr": " OR ".join(search_terms), "pageSize": 1000}
    studies = []
    while True:
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        studies += page["studies"]
        if not page.get("nextPageToken"):
            return studies
        params["pageToken"] = page["nextPageToken"]


def _date(value: str | None) -> date | None:
    """API dates are 'YYYY-MM-DD' or 'YYYY-MM'; the latter becomes the first of the month."""
    if not value:
        return None
    return date.fromisoformat(value if len(value) == 10 else f"{value}-01")


def parse_trial(study: dict) -> dict:
    p = study["protocolSection"]
    status = p.get("statusModule", {})
    return {
        "nct_id": p["identificationModule"]["nctId"],
        "title": p["identificationModule"].get("briefTitle"),
        "phase": p.get("designModule", {}).get("phases", []),
        "status": status.get("overallStatus"),
        "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        "enrollment": p.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
        "first_posted": _date(status.get("studyFirstPostDateStruct", {}).get("date")),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "completion_date": status.get("completionDateStruct", {}).get("date"),
        "interventions": [i["name"] for i in p.get("armsInterventionsModule", {}).get("interventions", [])],
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "has_results": study.get("hasResults", False),
        "last_updated": status.get("lastUpdatePostDateStruct", {}).get("date"),
    }


def _outcomes(items: list[dict]) -> str:
    return "\n".join(f"- {o['measure']} ({o.get('timeFrame', 'time frame not stated')})" for o in items)


def to_document(study: dict, aliases: dict[str, str]) -> dict:
    p = study["protocolSection"]
    trial = parse_trial(study)
    arms = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    candidates = {
        "Brief summary": p.get("descriptionModule", {}).get("briefSummary"),
        "Detailed description": p.get("descriptionModule", {}).get("detailedDescription"),
        "Primary outcomes": _outcomes(outcomes.get("primaryOutcomes", [])),
        "Secondary outcomes": _outcomes(outcomes.get("secondaryOutcomes", [])),
        "Arms": "\n".join(f"- {a['label']}: {a.get('description', '')}" for a in arms.get("armGroups", [])),
        "Eligibility": p.get("eligibilityModule", {}).get("eligibilityCriteria"),
    }
    sections = {name: text for name, text in candidates.items() if text}
    title = p["identificationModule"].get("officialTitle") or trial["title"]
    return {
        "doc_id": f"ctgov:{trial['nct_id']}",
        "source": "clinicaltrials.gov",
        "url": f"https://clinicaltrials.gov/study/{trial['nct_id']}",
        "title": title,
        "published_date": trial["first_posted"],
        "license": None,  # US government registry data
        "sections": sections,
        "metadata": {
            "drugs": matched_drugs(" ".join(trial["interventions"] + [title, trial["title"] or ""]), aliases),
            "nct_ids": [trial["nct_id"]],
            "phase": trial["phase"],
            "status": trial["status"],
            "year": trial["first_posted"].year if trial["first_posted"] else None,
        },
        "content_hash": content_hash(title, sections),
    }


def main() -> None:
    with connect() as conn:
        aliases = drug_aliases(conn)
        studies = fetch_studies(search_terms(aliases))
        print(f"fetched {len(studies)} studies")
        upsert_trials(conn, [parse_trial(s) for s in studies])
        counts = upsert_documents(conn, [to_document(s, aliases) for s in studies])
        print(f"trials upserted: {len(studies)}; documents: {counts}")


if __name__ == "__main__":
    main()
