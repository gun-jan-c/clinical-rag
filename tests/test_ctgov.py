from datetime import date

from clinical_rag.ingest.ctgov import parse_trial, to_document
from clinical_rag.ingest.drugs import matched_drugs

ALIASES = {"ly3502970": "orforglipron"}

# Trimmed shape of one API v2 study; values are placeholders.
STUDY = {
    "hasResults": False,
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Study of LY3502970",
                                 "officialTitle": "A Phase 3 Study of LY3502970 in Adults With Obesity"},
        "statusModule": {"overallStatus": "RECRUITING", "studyFirstPostDateStruct": {"date": "2024-05"},
                         "startDateStruct": {"date": "2024-06-01"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
        "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 100}},
        "armsInterventionsModule": {"interventions": [{"name": "LY3502970"}, {"name": "Placebo"}],
                                    "armGroups": [{"label": "Arm A", "description": "Daily dose"}]},
        "outcomesModule": {"primaryOutcomes": [{"measure": "Percent change in body weight",
                                                "timeFrame": "Week 72"}]},
        "descriptionModule": {"briefSummary": "Summary text"},
    },
}


def test_parse_trial_handles_month_only_dates_and_missing_fields():
    trial = parse_trial(STUDY)
    assert trial["first_posted"] == date(2024, 5, 1)
    assert trial["completion_date"] is None and trial["conditions"] == []
    assert trial["interventions"] == ["LY3502970", "Placebo"]


def test_document_tags_drug_by_code_name():
    doc = to_document(STUDY, ALIASES)
    assert doc["doc_id"] == "ctgov:NCT00000001"
    assert doc["metadata"]["drugs"] == ["orforglipron"]
    assert doc["sections"]["Primary outcomes"] == "- Percent change in body weight (Week 72)"
    assert "Eligibility" not in doc["sections"]  # empty sections are dropped


def test_document_tags_drug_named_only_in_brief_title():
    study = {"protocolSection": {**STUDY["protocolSection"],
                                 "identificationModule": {"nctId": "NCT00000002",
                                                          "briefTitle": "Response to Semaglutide",
                                                          "officialTitle": "An Observational Study"},
                                 "armsInterventionsModule": {}}}
    assert to_document(study, ALIASES)["metadata"]["drugs"] == ["semaglutide"]


def test_matched_drugs_finds_generic_names():
    assert matched_drugs("Tirzepatide vs Semaglutide", {}) == ["semaglutide", "tirzepatide"]
