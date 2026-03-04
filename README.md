# PRISMA per LiITA


**P**attern-based **R**ules for **I**ntent-driven **S**PARQL with **M**ultiple-resource **A**ssembly.

A transparent NL2SPARQL system for the [LiITA (Linking Italian)](https://liita.it) knowledge base.

## Overview

PRISMA is a pattern-based approach to translating natural language queries into SPARQL. Unlike end-to-end LLM-based systems, PRISMA uses the LLM **only for intent classification** while keeping SPARQL generation **deterministic and transparent**.

```
Natural Language Query
         │
         ▼
┌─────────────────────┐
│  Intent Analyzer    │  ◄── LLM (classification only)
│  (LLM-based)        │
└─────────────────────┘
         │
         ▼ Intent JSON
┌─────────────────────┐
│  Pattern            │  ◄── Deterministic rules
│  Orchestrator       │
└─────────────────────┘
         │
         ▼ Execution Plan
┌─────────────────────┐
│  Pattern            │  ◄── Template-based
│  Assembler          │
└─────────────────────┘
         │
         ▼
    SPARQL Query
```

## Features

- **Transparent Pipeline**: See exactly how each query is processed
- **Multi-Provider LLM Support**: Mistral, Anthropic, OpenAI, Gemini, Ollama
- **Pattern-Based Generation**: Guaranteed structural correctness
- **Multiple Resources**: CompL-it, Sentix, ELIta, Parmigiano, Sicilian dialects
- **Web Interface**: User-friendly Gradio UI

## Installation

```bash
# Clone the repository
git clone https://github.com/tonazzog/prisma-per-liita.git
cd prisma-per-liita

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Python API

```python
from prisma import Translator, create_llm_client

# Create an LLM client (supports: mistral, anthropic, openai, gemini, ollama)
client = create_llm_client(
    provider="mistral",
    api_key="your-api-key"  # Or set MISTRAL_API_KEY environment variable
)

# Create the translator
translator = Translator(client)

# Translate a query
result = translator.translate("What is the polarity of 'amore'?")

# Get the SPARQL query
print(result.sparql_query)

# Access intermediate results
print(f"Query type: {result.intent_dict['query_type']}")
print(f"Resources: {result.intent_dict['required_resources']}")
print(f"Steps: {result.total_steps}")
```

### Web Interface

```bash
# Run with default settings (Mistral)
python -m app.gradio_app

# Run with specific provider
python -m app.gradio_app --provider anthropic --model claude-sonnet-4-20250514

# Create a public link
python -m app.gradio_app --share
```

Then open http://localhost:7860 in your browser.

## Supported Query Types

| Query Type | Description | Example |
|------------|-------------|---------|
| `basic_lemma_lookup` | Simple LiITA queries | "How many nouns are in LiITA?" |
| `complit_semantic` | Semantic relations | "Find hyponyms of 'colore'" |
| `complit_definitions` | Definition search | "Words whose definition contains 'uccello'" |
| `dialect_translation` | Dialect translations | "Translate 'casa' to Parmigiano" |
| `sentix_polarity` | Sentiment analysis | "What is the polarity of 'amore'?" |
| `elita_emotion` | Emotion analysis | "What emotion is associated with 'felicità'?" |
| `multi_resource` | Combined queries | "Hyponyms of 'colore' with Sicilian translations" |

## LLM Providers

### Environment Variables

```bash
export MISTRAL_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Supported Providers

| Provider | Default Model | Package |
|----------|---------------|---------|
| Mistral | `mistral-large-latest` | `mistralai` |
| Anthropic | `claude-sonnet-4-20250514` | `anthropic` |
| OpenAI | `gpt-4o` | `openai` |
| Gemini | `gemini-1.5-pro` | `google-generativeai` |
| Ollama | `llama3.1` | `ollama` |

## Project Structure

```
prisma-per-liita/
├── prisma/                    # Main package
│   ├── __init__.py           # Package exports
│   ├── intent_analyzer.py    # LLM-based intent classification
│   ├── orchestrator.py       # Rule-based execution planning
│   ├── pattern_tools.py      # SPARQL pattern templates
│   ├── assembler.py          # Query assembly
│   └── translator.py         # Main entry point
│
├── shared/                   # Shared utilities
│   └── llm.py                # Multi-provider LLM interface
│
├── evaluation/               # Evaluation module
│   └── f1_evaluator.py       # F1 score computation engine
│
├── data/
│   └── test_dataset.json     # 100 annotated test cases
│
├── scripts/
│   ├── run_f1_evaluation.py  # Main evaluation entry point
│   └── replay_evaluation.py  # Re-score from existing reports
│
├── reports/                  # Generated evaluation reports (JSON)
│
├── app/                      # Web interface
│   └── gradio_app.py
│
├── examples/                 # Example scripts
│   └── basic_usage.py
│
└── tests/                    # Unit tests
```

## Architecture

### 1. Intent Analyzer (LLM)

The only component that uses an LLM. It classifies the query and extracts parameters:

```json
{
  "query_type": "sentix_polarity",
  "required_resources": ["liita_lemma_bank", "sentix"],
  "lemma": "amore",
  "semantic_relation": null,
  "complexity_score": 2
}
```

### 2. Pattern Orchestrator (Rules)

Deterministic rules that map intents to execution plans:

```
Step 1: liita_basic_query → ?lemma
Step 2: sentix_linking (depends on Step 1) → ?polarity
```

### 3. Pattern Assembler (Templates)

Composes SPARQL fragments with guaranteed structural correctness:

```sparql
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX marl: <http://www.gsi.upm.es/ontologies/marl/ns#>

SELECT ?lemma ?writtenRep ?polarity ?polarityValue
WHERE {
  ?lemma a ontolex:LexicalEntry ;
         ontolex:canonicalForm ?form .
  ?form ontolex:writtenRep ?writtenRep .
  FILTER(str(?writtenRep) = "amore") .

  ?sentixEntry ontolex:canonicalForm ?form ;
               marl:hasPolarity ?polarity ;
               marl:polarityValue ?polarityValue .
}
```

## Evaluation

### Metric: F1 Score on Answer Sets

PRISMA is evaluated with **F1 score on answer sets** — the standard metric for question-answering over knowledge bases. Rather than comparing SPARQL query strings, the evaluator executes both the gold query and the predicted query against the live LiITA endpoint and compares their result sets.

For each test case:

```
Precision = |gold ∩ predicted| / |predicted|   (fraction of returned answers that are correct)
Recall    = |gold ∩ predicted| / |gold|         (fraction of correct answers that were returned)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

The intersection is computed as a **multiset** (Counter): duplicate result rows are counted, so a query that returns the right answers but with incorrect cardinality is penalised.

#### Why F1 and not exact-match SPARQL?

String-level comparison of SPARQL queries is too strict for evaluating NL2SPARQL systems:

- **Variable renaming**: `?wr` and `?writtenRep` are semantically identical but string-different.
- **Clause ordering**: SPARQL has no required clause order; two syntactically different queries can be logically equivalent.
- **Alternative paths**: multiple valid queries may exist for the same natural language question, especially for multi-hop queries over LiITA's linked resources.

F1 on answer sets sidesteps all of these issues by measuring what actually matters: **whether the system returns the right answers**, regardless of how the query is written.

#### Variable mapping

PRISMA's predicted queries sometimes use different variable names than the gold queries (e.g. `?italianWord` vs `?italianWR`). The evaluator resolves this automatically through a four-phase matching strategy:

1. Exact name match within the same variable category (primary, aggregate, numeric)
2. Exact name match across categories
3. Longest common substring (threshold: 6 characters) — catches systematic renames
4. Positional fallback within category

Only **primary variables** (user-facing answer strings) are compared. Internal join variables (URIs, intermediate lemma nodes) are excluded.

### Test Dataset

The test set consists of **100 annotated questions** covering the full range of query patterns supported by PRISMA:

| Category | Count | Examples |
|---|---|---|
| `complex` | 56 | Multi-resource, SERVICE integration, dialect + emotion combined |
| `semantic_combined` | 29 | Semantic relations with filters, cross-resource lookups |
| `emotion` | 9 | ELIta emotion queries |
| `translation` | 6 | Dialect translation queries |

Each test case includes the NL question, the gold SPARQL query, and the set of answer variables to compare (split into primary, secondary, aggregate, and numeric categories).

### Running the Evaluation

The evaluation script translates every question in the test set and computes F1 against the gold results fetched from the LiITA endpoint.

#### Basic usage

```bash
# Anthropic Claude Sonnet 4.6 (recommended)
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6

# Mistral (with RPS rate limit for free tier: 1 req/s)
python scripts/run_f1_evaluation.py \
    --provider mistral \
    --delay 1.1

# OpenAI
python scripts/run_f1_evaluation.py \
    --provider openai \
    --model gpt-4.1-mini

# Local model via Ollama (no rate limiting needed)
python scripts/run_f1_evaluation.py \
    --provider ollama \
    --model llama3.2
```

The API key is read from the corresponding environment variable (`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) or passed explicitly with `--api-key`.

#### Rate limiting

For API providers with token-per-minute (TPM) quotas, use `--tpm-limit` to let the script pace calls automatically using a sliding 60-second window:

```bash
# Anthropic Tier 1 (30K ITPM)
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --tpm-limit 30000

# Anthropic Tier 2 (450K ITPM)
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --tpm-limit 450000
```

Each call to `claude-sonnet-4-6` consumes approximately **3,500 input tokens** (system prompt + 5-shot examples + question) and **~400 output tokens** (JSON intent object).

#### Cost estimate (`claude-sonnet-4-6`, 100 questions)

| | Tokens | Cost |
|---|---|---|
| Input ($3.00 / M) | ~348,000 | ~$1.04 |
| Output ($15.00 / M) | ~40,000 | ~$0.60 |
| **Standard total** | | **~$1.65** |
| **Batch API (50% off)** | | **~$0.82** |

#### Re-running failed cases

If some cases failed (e.g. due to a temporary API error), you can re-translate only those without repeating the full run:

```bash
python scripts/run_f1_evaluation.py \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --rerun-errors \
    --output reports/f1_report_prisma_anthropic_claude-sonnet-4-6.json
```

#### Re-scoring from an existing report

To re-evaluate predictions stored in a previous report against an updated gold dataset (without re-translating):

```bash
python scripts/replay_evaluation.py \
    reports/f1_report_prisma_anthropic_claude-sonnet-4-6.json
```

#### Output

Results are saved to `reports/f1_report_prisma_<provider>_<model>.json` and include aggregate metrics as well as per-case details (predicted SPARQL, variable mapping, precision/recall/F1, and any errors):

```
=======================================================
F1 EVALUATION RESULTS  —  PRISMA / anthropic / claude-sonnet-4-6
=======================================================
  Evaluated : 100
  Skipped   : 0  (gold query failed)
  Avg F1    : 0.6732
  Macro F1  : 0.6185
  Precision : 0.6632
  Recall    : 0.7300

  By category:
    complex                   avg F1=0.6866  (n=56)
    emotion                   avg F1=0.1895  (n=9)
    semantic_combined         avg F1=0.7643  (n=29)
    translation               avg F1=0.8333  (n=6)

  By pattern (top 10):
    LEXICAL_FORM                 avg F1=1.0000  (n=7)
    MORPHO_REGEX                 avg F1=0.9243  (n=13)
    SENSE_COUNT                  avg F1=0.8571  (n=14)
    TRANSLATION                  avg F1=0.8363  (n=29)
    SENSE_DEFINITION             avg F1=0.8106  (n=18)
    MULTI_TRANSLATION            avg F1=0.7942  (n=28)
    SERVICE_INTEGRATION          avg F1=0.7567  (n=48)
    META_GRAPH                   avg F1=0.6522  (n=23)
    EMOTION_LEXICON              avg F1=0.6100  (n=36)
    SEMANTIC_RELATION            avg F1=0.5724  (n=17)

  Score distribution:
    F1 = 1.00 (perfect): 46
    F1 = 0.00          : 24
    Translation errors : 6
```

## License

MIT License - see [LICENSE](LICENSE) for details.

