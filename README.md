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
│   ├── llm.py                # Multi-provider LLM interface
│   ├── intent_analyzer.py    # LLM-based intent classification
│   ├── orchestrator.py       # Rule-based execution planning
│   ├── pattern_tools.py      # SPARQL pattern templates
│   ├── assembler.py          # Query assembly
│   └── translator.py         # Main entry point
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

## License

MIT License - see [LICENSE](LICENSE) for details.

