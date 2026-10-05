"""Section templates (ProjectSpec.md sections 2, 3 and 5.7): the writing rules, and per LLM section what to search
for, what Cohere ranks the candidates by, and what to write. `{drug}` is filled in per drug."""

from clinical_rag.schemas import SectionKey

SECTION_TITLES: dict[SectionKey, str] = {
    "pipeline": "Development pipeline by phase",
    "efficacy": "Key efficacy results",
    "safety": "Safety and tolerability",
    "competitive_positioning": "Competitive positioning",
    "evidence_gaps": "Open questions and evidence gaps",
}

RULES = """1. Use only the provided sources. Each source is labeled with its id and content type (text, table, figure).
2. Every claim must cite one or more source ids that directly support it.
3. Copy numbers exactly as written in the source (values, units, confidence intervals, timepoints, dose arms).
4. If information the question asks for is not in the sources, add it to not_found. Never fill gaps from general \
knowledge, even when you know the answer.
5. When citing a figure, describe only what the caption or description states; never estimate values from a figure.
6. Neutral, scientific tone. No promotional language, no treatment recommendations, no comparative superiority \
claims unless a head-to-head trial in the sources states it."""

ANSWER_PROMPT = f"You answer questions about obesity drugs using only the sources provided.\n\n{RULES}"


def section_prompt(key: SectionKey, instructions: str) -> str:
    return (f'You write the "{SECTION_TITLES[key]}" section of a competitive landscape brief on obesity drugs, '
            f"using only the sources provided.\n\nWhat to cover: {instructions}\n\n{RULES}")


TEMPLATES: dict[SectionKey, dict] = {
    "efficacy": {
        "queries": ["{drug} percent change in body weight primary endpoint",
                    "{drug} proportion of participants achieving 5% 10% 15% weight reduction",
                    "{drug} phase 3 trial results obesity"],
        "rerank": "Key efficacy results of {drug} in obesity: change in body weight and responder rates by dose",
        "write": "For each drug, the main weight-loss results: trial name, dose, timepoint, comparator, mean change "
                 "in body weight and the share of participants reaching weight-loss thresholds. Say when a result "
                 "comes from a subgroup or post hoc analysis rather than the trial's main analysis.",
    },
    "safety": {
        "queries": ["{drug} adverse reactions nausea vomiting diarrhea constipation",
                    "{drug} discontinuation due to adverse events",
                    "{drug} serious adverse events warnings and precautions"],
        "rerank": "Safety and tolerability of {drug}: common adverse reactions, discontinuations, serious risks",
        "write": "For each drug, the most common adverse reactions with their rates, discontinuations due to "
                 "adverse events, and serious risks or warnings.",
    },
    "competitive_positioning": {
        "queries": ["{drug} dosing and administration route frequency",
                    "{drug} approval status indication obesity",
                    "{drug} head-to-head comparison"],
        "rerank": "How {drug} is given and where it stands: route, dosing frequency, approval status, comparisons",
        "write": "For each drug, the route and dosing frequency, approval status for weight management, and any "
                 "head-to-head comparisons with other drugs reported in the sources.",
    },
}

EVIDENCE_GAPS_WRITE = ("For each drug, what is not yet known: what the ongoing trials in the sources are testing "
                       "(status, phase, enrollment), and which open questions remain. Put open questions that no "
                       "source answers in not_found, including the ones listed from other sections.")
