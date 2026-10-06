# RAG in Aurelia

"RAG" is a family of techniques, not one. Aurelia implements thirteen over the same synthetic record so they can be compared directly (Ask → *Compare strategies*, or the **RAG Lab** view).

| Strategy | Idea | Use it when | Weak at |
|---|---|---|---|
| **Adaptive router** | Classify the question, then pick a strategy and explain why | Default | Inherits the weakness of whatever it picks |
| **BM25** | Lexical match | Question reuses record wording | Paraphrase, lay terms |
| **Dense (naive)** | Embed and cosine‑match | Paraphrase | Exact identifiers, numbers |
| **Hybrid** | BM25 + dense merged by Reciprocal Rank Fusion | General default | Time and aggregation |
| **Multi‑query** | Decompose compound questions, rewrite each, fuse | "X and also Y?" | Cost: several retrievals |
| **HyDE** | Write the passage that *would* answer, search with it | Vague/lay questions | Template quality (LLM optional) |
| **Rerank** | Wide pool → re‑score on coverage, phrase, type fit, abnormality, recency | Right chunk is in the pool but not on top | Heuristic, not learned |
| **MMR** | Relevance minus redundancy | Broad questions that return five near‑identical rows | Can drop a second relevant duplicate |
| **Small‑to‑big** | Match small chunks, return neighbours (same test history, same visit) | Value needs context | Longer evidence |
| **Temporal** | Parse "latest", "last 6 months", "in 2025"; filter/rank by date | Anything about recency | Questions with no time cue |
| **Corrective (CRAG)** | Grade own evidence; rewrite and retry; abstain if still weak | Safety‑critical "is it documented?" | May abstain on poorly worded valid questions |
| **Graph / cohort** | Parse constraints → exact evaluation on a knowledge graph | "Which patients on X with Y?" | Free‑text questions; unrecognised phrasing |
| **Agentic** | Plan tool calls (search, trends, alerts, meds, cohort, notes), run, reflect | Multi‑part analytical questions | Slowest; planner is rule‑based |

## Supporting techniques

- **Grading** – coverage of the question's content words (synonym‑aware) in the top three evidence chunks. ≥ 0.60 high, ≥ 0.34 medium, else low → retry or abstain.
- **Citation verification** – every answer line must cite evidence; numbers in the line must appear in a cited chunk and ≥ 50 % of its content tokens must overlap. Lines marked `≈` are model‑derived (counts, forecasts) and excluded.
- **Conversation memory** – a short follow‑up carries over modifiers (trend, latest, abnormal, time window) from the previous question.
- **Patient scoping** – filtering happens *before* ranking, so a scoped query cannot return another patient's chunks.

## Beyond RAG

Similar‑patient search (cosine over a clinical vector), k‑means cohorts with a 3D PCA map, OLS lab forecasts with a prediction interval (needs ≥ 4 points), robust z‑score anomaly detection, negation‑aware note NER, rule‑based safety alerts, explainable risk score.

## Honest evaluation

Run `python -m eval.benchmark`. See the README table and its caveats: template‑generated synthetic questions; faithfulness is high largely because answers are extractive; paraphrased cohort questions remain the weak spot.
