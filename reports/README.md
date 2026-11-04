# Evaluation reports

Reports in this folder are the ones used in the thesis: canonical gold
(`data/test_dataset.json`, hypolemmas included), unified F1 evaluator,
`mistral-large-4-0`, run on 4 November 2026.

| File | Avg F1 | Notes |
| --- | --- | --- |
| `f1_report_prisma_mistral_mistral-large-4-0.json` | 0.740 | After the filter fix (regex escaping, flags argument inside `SERVICE`) |

Each run has three files: `.json` (report), `.calls.jsonl` (one line per LLM call
with token usage) and `.log` (console output).

## archive/

Not used for the thesis results.

- `superseded/` – same setup, earlier code: `..._prefix` is the run before the
  filter fix (avg F1 0.674, 11 endpoint errors).
- `2026-03/` – March 2026 runs (Claude Sonnet 4.6) on the older gold and evaluator;
  not comparable with the reports above.
