# Norland legal benchmark collection

A small test collection written for this project, in the standard BEIR format.

- `corpus.jsonl`: 61 articles from five **fictional** statutes of the imaginary "Republic of Norland" (Labour Code, Personal Data Protection Act, Consumer Rights Act, Residential Tenancy Act, Companies Act). They are modelled on the kinds of rules found in European law, but they are **not real law** and must not be used as legal information.
- `queries.jsonl`: 56 questions phrased the way non-lawyers ask them. `metadata.split` alternates between `train` (used only to fit the reranker) and `test` (held out for reporting).
- `qrels.tsv`: graded relevance judgements (2 = answers the question, 1 = related).

Each article's metadata has `act`, `act_code`, `article` and `year`, which are used to test filters.
