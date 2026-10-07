# Evaluation report

Golden set: 31 questions, 14 types. Chosen configuration: **section/small/hybrid/rerank on +gen**, the pipeline the app runs (hybrid retrieval, 50-candidate pool, Cohere rerank to the top 8).

## Results across configurations

Dev split, 50-candidate pool. Recall is document level; strict unless noted.

| configuration | recall@5 | recall@10 | MRR | lenient@10 | citations | grounded | not-found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| section/small/hybrid/rerank off | 53.6% | 64.3% | 44.5% | 78.6% | n/a | n/a | n/a |
| section/small/dense/rerank off | 53.6% | 71.4% | 47.7% | 71.4% | n/a | n/a | n/a |
| section/small/keyword/rerank off | 21.4% | 50.0% | 16.7% | 71.4% | n/a | n/a | n/a |
| section/large/hybrid/rerank off | 57.1% | 71.4% | 46.0% | 78.6% | n/a | n/a | n/a |
| fixed/small/hybrid/rerank off | 53.6% | 60.7% | 41.9% | 85.7% | n/a | n/a | n/a |
| fixed/small/hybrid/rerank on | 67.9% | 67.9% | 62.5% | 85.7% | n/a | n/a | n/a |
| section/small/hybrid/rerank on +gen | 78.6% | 85.7% | 65.9% | 85.7% | 97.4% | 97.4% | 100.0% |

## Table experiment: section/markdown vs fixed/flattened

- **section/markdown**: table-question recall@10 100.0% (n=3); overall recall@10 85.7%.
- **fixed/flattened**: table-question recall@10 100.0% (n=3); overall recall@10 67.9%.

Section chunks keep each table whole with its header, so the model reads the right row; fixed chunks flatten tables into text and can split a table across slices. See the decision log (gap B) for why some label rows are still found only through the table, not its summary.

## Failure examples

From the production run (`section/small/hybrid/rerank on +gen`):

- **q19 [safety]** — What were the most common adverse events with tirzepatide in SURMOUNT-1?
  - expected a relevant doc in ['pubmed:35658024']; top 3 retrieved: ['epmc:PMC12933229', 'epmc:PMC11885085', 'epmc:PMC13448863']
  - The SURMOUNT-1 abstract mentions adverse events in a single sentence, so for an 'adverse events' query it is out-competed by papers written entirely about tirzepatide safety and falls past the 50-candidate pool. The same paper ranks fine for the weight-loss wording (q01): retrieval is phrasing-sensitive, and the multi-query brief path is more robust than a single question.
- **q23 [population]** — What BMI was required to enter STEP 1?
  - expected a relevant doc in ['ctgov:NCT03548935']; top 3 retrieved: ['epmc:PMC7318657', 'epmc:PMC12673442', 'epmc:PMC13375533']
  - Fix D put 'STEP 1' into the record's text, but it still does not reach the 50-candidate pool for this phrasing: 'step' is a common token (STEP 2-8 and any 'step' elsewhere) and the BMI-eligibility wording is generic across ~313 trials, while an exact acronym match gets no extra weight. Still a retrieval miss; fix D was necessary but not sufficient. Fix D did lift other acronym questions (dev MRR 59.9% -> 65.9%, key_facts 84.6% -> 92.3%). A stronger fix would boost the acronym/title or apply a trial filter.

---
Regenerate with `uv run python -m clinical_rag.eval.report` after new runs.
