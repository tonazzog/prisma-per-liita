"""
PRISMA Intent Analyzer
======================

LLM-based intent analysis system that converts natural language queries
into structured Intent objects for the pattern-based SPARQL generator.

This is the ONLY component that uses LLM - and it has a focused task:
classify the query and extract parameters (not generate SPARQL).

Supports multiple LLM providers via the llm module:
- Mistral
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Ollama (local models)

Part of PRISMA per LiITA - Pattern-based Rules for Intent-driven SPARQL
with Multiple-resource Assembly for the LiITA knowledge base.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict


# ============================================================================
# Intent Analyzer Prompt Template
# ============================================================================

INTENT_ANALYZER_SYSTEM_PROMPT = """You are an expert intent analyzer for the LiITA Knowledge Base, a linguistic resource for Italian that links multiple lexical resources.

Your ONLY task is to analyze natural language queries about Italian linguistics and produce structured JSON output. You do NOT generate SPARQL queries.

# AVAILABLE RESOURCES

1. **LiITA Lemma Bank** (Local)
   - Italian lemmas with part-of-speech tags
   - Written representations
   - Basic filtering capabilities

2. **CompL-it** (External - requires SERVICE clause)
   - Computational lexicon with 101K+ entries
   - Natural language definitions
   - Semantic relations: hypernyms, hyponyms, meronyms, holonyms, synonyms
   - 137 types of semantic relations
   - Usage examples

3. **Parmigiano Lexicon** (Local)
   - Italian ↔ Parmigiano dialect translations
   - Pattern-based searching available
   - Linked via LiITA lemmas

4. **Sicilian Lexicon** (Local)
   - Italian ↔ Sicilian dialect translations
   - Pattern-based searching available
   - Linked via LiITA lemmas

5. **Sentix** (Local)
   - Sentiment/polarity lexicon for Italian (63,660 entries)
   - Links to LiITA lemmas via ontolex:canonicalForm
   - Provides polarity type (Positive/Negative/Neutral) via marl:hasPolarity
   - Provides polarity value (-1 to +1) via marl:hasPolarityValue
   - Use for sentiment analysis queries

6. **ELIta** (Local)
   - Emotion lexicon for Italian
   - Links to LiITA lemmas via ontolex:canonicalForm
   - Provides emotion classification via elita:HasEmotion
   - Emotions: Gioia (Joy), Tristezza (Sadness), Rabbia (Anger), Paura (Fear), Disgusto (Disgust), Sorpresa (Surprise)
   - Use for emotion analysis queries

# QUERY TYPES

Classify queries into ONE of these types:

1. **basic_lemma_lookup**
   - Simple queries on LiITA only
   - Examples: "How many nouns?", "Find words starting with X"

2. **complit_definitions**
   - Search CompL-it by definition content
   - Examples: "Words whose definition contains X", "Definitions starting with Y"

3. **complit_semantic**
   - Navigate semantic relations in CompL-it
   - Examples: "Hyponyms of X", "What are meronyms of Y", "Antonyms of Z"

4. **complit_relation_check**
   - Check if a specific relation exists between two lemmas
   - Examples: "Are 'cane' and 'animale' related?", "Is 'rosso' a hyponym of 'colore'?"

5. **complit_word_sense_lookup**
   - Look up a word by its written form and retrieve all its senses with definitions
   - Examples: "Find all senses of the word 'vita'", "What are the meanings of 'piano'?"

6. **dialect_pattern_search**
   - Pattern-based search in dialect (Sicilian or Parmigiano)
   - Examples: "Sicilian words ending in ìa", "Parmigiano words ending in 'u'"

7. **dialect_translation**
   - Translate specific words between Italian and dialects
   - Examples: "Translate casa to Parmigiano"

8. **multi_resource**
   - Complex queries combining CompL-it with dialects
   - Examples: "Hyponyms of X with Parmigiano translations"

9. **sentix_polarity**
   - Queries about sentiment/polarity of words using Sentix
   - Examples: "What is the polarity of 'amore'?", "Find positive words", "Sentiment of 'guerra'"

10. **elita_emotion**
   - Queries about emotions associated with words using ELIta
   - Examples: "What emotion is associated with 'felicità'?", "Find words with joy emotion"

11. **affective_multi_resource**
   - Complex queries combining Sentix/ELIta with other resources
   - Examples: "Find positive words with their definitions", "Emotions of color hyponyms"

# SEMANTIC RELATIONS

When identifying semantic relations, use these exact values:
- **hyponym**: X is a type of Y (e.g., "cat" is a hyponym of "animal")
- **hypernym**: X is a category containing Y (e.g., "animal" is a hypernym of "cat")
- **meronym**: X is part of Y (e.g., "wheel" is a meronym of "car")
- **holonym**: X contains Y as part (e.g., "car" is a holonym of "wheel")
- **synonym**: X has similar meaning to Y

# OUTPUT FORMAT

You must output valid JSON with this exact structure:

```json
{
  "query_type": "one of: basic_lemma_lookup | complit_definitions | complit_semantic | complit_relation_check | complit_word_sense_lookup | dialect_pattern_search | dialect_translation | multi_resource | sentix_polarity | elita_emotion | affective_multi_resource",
  "required_resources": ["array of: liita_lemma_bank | complit | parmigiano | sicilian | sentix | elita"],
  "lemma": "specific lemma mentioned (or null)",
  "lemma_b": "second lemma for relation checking queries (or null)",
  "pos": "part of speech: noun | verb | adjective | adverb (or null)",
  "definition_pattern": "text pattern to match in definitions (or null)",
  "pattern_type": "starts_with | contains | ends_with | regex (or null)",
  "written_form_pattern": "regex pattern for written forms (or null)",
  "semantic_relation": "hyponym | hypernym | meronym | holonym | synonym | antonym (or null)",
  "filters": [],
  "aggregation": {
    "type": "count | group_concat | distinct (or null)",
    "aggregate_var": "variable to aggregate",
    "group_by_vars": ["array of variables"],
    "separator": ", ",
    "order_by": {
      "var": "variable to sort by",
      "direction": "ASC | DESC"
    }
  },
  "retrieve_definitions": true,
  "retrieve_examples": false,
  "include_italian_written_rep": true,
  "complexity_score": 1-5,
  "user_query": "original query text",
  "reasoning": "brief explanation of your classification"
}
```

# CRITICAL RULES

1. **Always output valid JSON** - no markdown, no explanations outside JSON
2. **Use exact enum values** - no variations or synonyms
3. **Aggregation only when EXPLICITLY requested** - Set aggregation to null unless the user explicitly asks for:
   - "count" / "how many" → use count aggregation
   - "group" / "grouped" → use group_concat aggregation
   - DO NOT add aggregation just for "readability" or to be "helpful"
   - Simple listing queries (e.g., "Find hyponyms of X") do NOT need aggregation
4. **Be conservative with complexity_score** - prefer lower scores
5. **Include reasoning** - helps debugging

# EXAMPLES

See the examples section below for correct classifications."""


# ============================================================================
# Few-Shot Examples
# ============================================================================

INTENT_EXAMPLES = [
    {
        "query": "How many nouns are in LiITA?",
        "intent": {
            "query_type": "basic_lemma_lookup",
            "required_resources": ["liita_lemma_bank"],
            "lemma": None,
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": {
                "type": "count",
                "aggregate_var": None,
                "group_by_vars": [],
                "separator": None,
                "order_by": None
            },
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 1,
            "user_query": "How many nouns are in LiITA?",
            "reasoning": "Simple count aggregation on LiITA lemmas filtered by POS"
        }
    },
    {
        "query": "How many Parmigiano nouns are there?",
        "intent": {
            "query_type": "dialect_pattern_search",
            "required_resources": ["parmigiano"],
            "lemma": None,
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": ".*",
            "semantic_relation": None,
            "filters": [],
            "aggregation": {
                "type": "count",
                "aggregate_var": None,
                "group_by_vars": [],
                "separator": None,
                "order_by": None
            },
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 2,
            "user_query": "How many Parmigiano nouns are there?",
            "reasoning": "Count dialect lemmas - only parmigiano resource needed, NOT liita_lemma_bank. Use dialect_pattern_search with wildcard pattern."
        }
    },
    {
        "query": "How many Sicilian verbs are there?",
        "intent": {
            "query_type": "dialect_pattern_search",
            "required_resources": ["sicilian"],
            "lemma": None,
            "pos": "verb",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": ".*",
            "semantic_relation": None,
            "filters": [],
            "aggregation": {
                "type": "count",
                "aggregate_var": None,
                "group_by_vars": [],
                "separator": None,
                "order_by": None
            },
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 2,
            "user_query": "How many Sicilian verbs are there?",
            "reasoning": "Count dialect lemmas - only sicilian resource needed. Use dialect_pattern_search with wildcard pattern and count aggregation."
        }
    },
    {
        "query": "What is the polarity of the word 'amore'?",
        "intent": {
            "query_type": "sentix_polarity",
            "required_resources": ["liita_lemma_bank", "sentix"],
            "lemma": "amore",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "What is the polarity of the word 'amore'?",
            "reasoning": "Sentiment/polarity lookup for a specific lemma using Sentix"
        }
    },
    {
        "query": "What emotion is associated with 'felicità'?",
        "intent": {
            "query_type": "elita_emotion",
            "required_resources": ["liita_lemma_bank", "elita"],
            "lemma": "felicità",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "What emotion is associated with 'felicità'?",
            "reasoning": "Emotion lookup for a specific lemma using ELIta"
        }
    },
    {
        "query": "Are 'cane' and 'animale' related?",
        "intent": {
            "query_type": "complit_relation_check",
            "required_resources": ["complit"],
            "lemma": "cane",
            "lemma_b": "animale",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 2,
            "user_query": "Are 'cane' and 'animale' related?",
            "reasoning": "Check if any semantic relation exists between two lemmas in CompL-it"
        }
    },
    {
        "query": "Is 'rosso' a hyponym of 'colore'?",
        "intent": {
            "query_type": "complit_relation_check",
            "required_resources": ["complit"],
            "lemma": "rosso",
            "lemma_b": "colore",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "hyponym",
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 2,
            "user_query": "Is 'rosso' a hyponym of 'colore'?",
            "reasoning": "Check specific hyponym relation between two lemmas"
        }
    },
    {
        "query": "Find Italian words in CompL-it whose definition starts with 'uccello'",
        "intent": {
            "query_type": "complit_definitions",
            "required_resources": ["complit"],
            "lemma": None,
            "pos": None,
            "definition_pattern": "uccello",
            "pattern_type": "starts_with",
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "Find Italian words in CompL-it whose definition starts with 'uccello'",
            "reasoning": "Definition search in CompL-it with pattern matching"
        }
    },
    {
        "query": "What are the hyponyms of 'colore' in CompL-it?",
        "intent": {
            "query_type": "complit_semantic",
            "required_resources": ["complit", "liita_lemma_bank"],
            "lemma": "colore",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "hyponym",
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "What are the hyponyms of 'colore' in CompL-it?",
            "reasoning": "Semantic relation navigation (hyponyms) for specific lemma. NO aggregation needed - just list the hyponyms."
        }
    },
    {
        "query": "Find the hyponyms of 'animale'",
        "intent": {
            "query_type": "complit_semantic",
            "required_resources": ["complit", "liita_lemma_bank"],
            "lemma": "animale",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "hyponym",
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "Find the hyponyms of 'animale'",
            "reasoning": "Simple semantic query - list hyponyms without aggregation. User did NOT ask for grouping or counting."
        }
    },
    {
        "query": "Find all senses of the word 'vita'",
        "intent": {
            "query_type": "complit_word_sense_lookup",
            "required_resources": ["complit"],
            "lemma": "vita",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "Find all senses of the word 'vita'",
            "reasoning": "Word sense lookup in CompL-it - retrieve all senses and definitions for a specific lemma"
        }
    },
    {
        "query": "What are the meanings of the noun 'piano'?",
        "intent": {
            "query_type": "complit_word_sense_lookup",
            "required_resources": ["complit"],
            "lemma": "piano",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "What are the meanings of the noun 'piano'?",
            "reasoning": "Word sense lookup with POS filter - retrieve all noun senses and definitions"
        }
    },
    {
        "query": "Find words whose definition starts with 'uccello' and show their Parmigiano translations",
        "intent": {
            "query_type": "multi_resource",
            "required_resources": ["complit", "liita_lemma_bank", "parmigiano"],
            "lemma": None,
            "pos": None,
            "definition_pattern": "uccello",
            "pattern_type": "starts_with",
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 4,
            "user_query": "Find words whose definition starts with 'uccello' and show their Parmigiano translations",
            "reasoning": "Multi-resource query: CompL-it definition search + Parmigiano translation"
        }
    },
    {
        "query": "Find hyponyms of 'colore' with Parmigiano translations",
        "intent": {
            "query_type": "multi_resource",
            "required_resources": ["complit", "liita_lemma_bank", "parmigiano"],
            "lemma": "colore",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "hyponym",
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 4,
            "user_query": "Find hyponyms of 'colore' with Parmigiano translations",
            "reasoning": "Multi-resource: semantic relations + dialect translation. NO aggregation - user just wants a list."
        }
    },
    {
        "query": "Count how many hyponyms 'animale' has",
        "intent": {
            "query_type": "complit_semantic",
            "required_resources": ["complit", "liita_lemma_bank"],
            "lemma": "animale",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "hyponym",
            "filters": [],
            "aggregation": {
                "type": "count",
                "aggregate_var": None,
                "group_by_vars": [],
                "separator": None,
                "order_by": None
            },
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 3,
            "user_query": "Count how many hyponyms 'animale' has",
            "reasoning": "User explicitly asked to COUNT - use count aggregation"
        }
    },
    {
        "query": "Find Sicilian nouns ending in 'ìa' and their Italian translations",
        "intent": {
            "query_type": "dialect_pattern_search",
            "required_resources": ["sicilian", "liita_lemma_bank"],
            "lemma": None,
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": "ìa$",
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "Find Sicilian nouns ending in 'ìa' and their Italian translations",
            "reasoning": "Pattern-based Sicilian search with Italian linking. NO aggregation - just list results."
        }
    },
    {
        "query": "Find Parmigiano nouns ending in 'u' and their Italian translations",
        "intent": {
            "query_type": "dialect_pattern_search",
            "required_resources": ["parmigiano", "liita_lemma_bank"],
            "lemma": None,
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": "u$",
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "Find Parmigiano nouns ending in 'u' and their Italian translations",
            "reasoning": "Pattern-based Parmigiano search with Italian linking. NO aggregation - just list results."
        }
    },
    {
        "query": "Find lemmas starting with 'infra'",
        "intent": {
            "query_type": "basic_lemma_lookup",
            "required_resources": ["liita_lemma_bank"],
            "lemma": None,
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": "^infra",
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 1,
            "user_query": "Find lemmas starting with 'infra'",
            "reasoning": "Simple LiITA pattern search with regex"
        }
    },
    {
        "query": "What are the meronyms of 'giorno'?",
        "intent": {
            "query_type": "complit_semantic",
            "required_resources": ["complit", "liita_lemma_bank"],
            "lemma": "giorno",
            "pos": "noun",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": "meronym",
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "What are the meronyms of 'giorno'?",
            "reasoning": "Semantic relation query (meronyms) in CompL-it"
        }
    },
    {
        "query": "Show me nouns in CompL-it with definitions containing 'colore'",
        "intent": {
            "query_type": "complit_definitions",
            "required_resources": ["complit"],
            "lemma": None,
            "pos": "noun",
            "definition_pattern": "colore",
            "pattern_type": "contains",
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "Show me nouns in CompL-it with definitions containing 'colore'",
            "reasoning": "Definition search with 'contains' pattern and POS filter"
        }
    },
    {
        "query": "Count verbs in LiITA",
        "intent": {
            "query_type": "basic_lemma_lookup",
            "required_resources": ["liita_lemma_bank"],
            "lemma": None,
            "pos": "verb",
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": {
                "type": "count",
                "aggregate_var": None,
                "group_by_vars": [],
                "separator": None,
                "order_by": None
            },
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": False,
            "complexity_score": 1,
            "user_query": "Count verbs in LiITA",
            "reasoning": "Simple count with POS filter"
        }
    },
    {
        "query": "The polarity of the word 'bello'",
        "intent": {
            "query_type": "sentix_polarity",
            "required_resources": ["liita_lemma_bank", "sentix"],
            "lemma": "bello",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "The polarity of the word 'bello'",
            "reasoning": "Sentiment/polarity lookup for a specific lemma using Sentix"
        }
    },
    {
        "query": "Find the sentiment of 'guerra'",
        "intent": {
            "query_type": "sentix_polarity",
            "required_resources": ["liita_lemma_bank", "sentix"],
            "lemma": "guerra",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "Find the sentiment of 'guerra'",
            "reasoning": "Sentiment/polarity lookup using Sentix - 'sentiment' indicates polarity query"
        }
    },
    {
        "query": "Find words associated with joy",
        "intent": {
            "query_type": "elita_emotion",
            "required_resources": ["liita_lemma_bank", "elita"],
            "lemma": None,
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [{"type": "emotion", "value": "Gioia"}],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 2,
            "user_query": "Find words associated with joy",
            "reasoning": "Search for lemmas with specific emotion (Gioia/Joy) using ELIta"
        }
    },
    {
        "query": "What is the polarity and emotion of 'amore'?",
        "intent": {
            "query_type": "affective_multi_resource",
            "required_resources": ["liita_lemma_bank", "sentix", "elita"],
            "lemma": "amore",
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 3,
            "user_query": "What is the polarity and emotion of 'amore'?",
            "reasoning": "Combined sentiment and emotion lookup using both Sentix and ELIta"
        }
    },
    {
        "query": "Find positive words with their definitions",
        "intent": {
            "query_type": "affective_multi_resource",
            "required_resources": ["liita_lemma_bank", "sentix", "complit"],
            "lemma": None,
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [{"type": "polarity", "value": "Positive"}],
            "aggregation": None,
            "retrieve_definitions": True,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 4,
            "user_query": "Find positive words with their definitions",
            "reasoning": "Multi-resource query combining Sentix polarity filter with CompL-it definitions"
        }
    }
]


# ============================================================================
# Intent Analyzer Class
# ============================================================================

class IntentAnalyzer:
    """
    LLM-based intent analyzer for LiITA queries.

    This is the ONLY component that uses an LLM. Its task is focused:
    classify the query and extract parameters (not generate SPARQL).

    Example usage with the multi-provider LLM module:

        from llm import create_llm_client

        # Create an LLM client (supports: mistral, anthropic, openai, gemini, ollama)
        client = create_llm_client(
            provider="anthropic",
            api_key="your-api-key",
            model="claude-sonnet-4-20250514"
        )

        # Create the analyzer
        analyzer = IntentAnalyzer(client)

        # Analyze a query
        intent_dict, warnings = analyzer.analyze("What are the hyponyms of 'colore'?")
    """

    def __init__(self, llm_client):
        """
        Initialize with an LLM client.

        Args:
            llm_client: Client that implements BaseLLM interface with
                       complete(prompt, system, temperature, max_tokens) method.
                       Use create_llm_client() from llm.py to create one.
        """
        self.llm_client = llm_client
        self.system_prompt = INTENT_ANALYZER_SYSTEM_PROMPT
        self.examples = INTENT_EXAMPLES
    
    def analyze(self, user_query: str, include_examples: int = 5) -> Tuple[Dict, List[str]]:
        """
        Analyze a user query and produce an Intent object.
        
        Args:
            user_query: Natural language query from user
            include_examples: Number of few-shot examples to include (0-11)
            
        Returns:
            (intent_dict, warnings) tuple
            - intent_dict: Parsed intent as dictionary
            - warnings: List of validation warnings
        """
        
        # Build the user prompt with examples
        user_prompt = self._build_user_prompt(user_query, include_examples)
        
        # Call LLM
        #try:
        response = self.llm_client.complete(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=0.0  # Deterministic for classification
        )
        
        # Parse JSON response
        intent_dict, parse_warnings = self._parse_response(response)
        print(intent_dict)
        # Validate intent
        validation_warnings = self._validate_intent(intent_dict)
        
        # Ensure user_query is set
        if 'user_query' not in intent_dict or not intent_dict['user_query']:
            intent_dict['user_query'] = user_query
        
        all_warnings = parse_warnings + validation_warnings
        
        return intent_dict, all_warnings
            
        #except Exception as e:
            # Return a default intent on error
            #return self._create_fallback_intent(user_query, str(e)), [f"LLM error: {str(e)}"]
    
    def _build_user_prompt(self, user_query: str, num_examples: int) -> str:
        """Build the user prompt with few-shot examples."""

        lines = []

        # Add examples if requested
        if num_examples > 0:
            lines.append("# CLASSIFICATION EXAMPLES\n")

            for i, example in enumerate(self.examples[:num_examples], 1):
                lines.append(f"## Example {i}")
                lines.append(f"Query: \"{example['query']}\"")
                lines.append(f"Intent JSON:")
                lines.append("```json")
                lines.append(json.dumps(example['intent'], indent=2))
                lines.append("```\n")

        # Add the actual query
        lines.append("# QUERY TO ANALYZE\n")
        lines.append(f"Query: \"{user_query}\"")
        lines.append("\nProvide the Intent JSON (no markdown, just JSON):")

        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> Tuple[Dict, List[str]]:
        """Parse LLM response into intent dictionary."""
        
        warnings = []
        
        # Try to extract JSON from response
        # Handle cases where LLM wraps JSON in markdown
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
        
        # Try to parse JSON
        try:
            intent_dict = json.loads(response)
            return intent_dict, warnings
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error: {str(e)}")
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    intent_dict = json.loads(json_match.group())
                    warnings.append("Extracted JSON from text response")
                    return intent_dict, warnings
                except json.JSONDecodeError:
                    pass
            
            # Return empty dict if parsing fails
            return {}, warnings
    
    def _validate_intent(self, intent_dict: Dict) -> List[str]:
        """Validate intent dictionary and return warnings."""
        
        warnings = []
        
        if not intent_dict:
            warnings.append("Empty intent dictionary")
            return warnings
        
        # Check required fields
        required_fields = ['query_type', 'required_resources']
        for field in required_fields:
            if field not in intent_dict:
                warnings.append(f"Missing required field: {field}")
        
        # Validate query_type
        valid_query_types = [
            'basic_lemma_lookup', 'complit_definitions', 'complit_semantic',
            'complit_relation_check', 'complit_word_sense_lookup', 'dialect_pattern_search',
            'dialect_translation', 'multi_resource', 'sentix_polarity', 'elita_emotion',
            'affective_multi_resource'
        ]
        if 'query_type' in intent_dict:
            if intent_dict['query_type'] not in valid_query_types:
                warnings.append(f"Invalid query_type: {intent_dict['query_type']}")
        
        # Validate resources
        valid_resources = ['liita_lemma_bank', 'complit', 'parmigiano', 'sicilian', 'sentix', 'elita']
        # Normalize resource names (handle common variations from LLM)
        resource_normalization = {
            'siciliano': 'sicilian',  # Italian word for Sicilian
            'parmigiano': 'parmigiano',  # Already correct
            'liita': 'liita_lemma_bank',
            'liita_lemma_bank': 'liita_lemma_bank',
            'complit': 'complit',
            'sentix': 'sentix',
            'elita': 'elita'
        }
        
        if 'required_resources' in intent_dict:
            for resource in intent_dict.get('required_resources', []):
                # Normalize the resource name before validation
                normalized_resource = resource_normalization.get(resource.lower(), resource.lower())
                if normalized_resource not in valid_resources:
                    warnings.append(f"Invalid resource: {resource} (normalized: {normalized_resource})")
        
        # Validate semantic_relation if present
        valid_relations = ['hyponym', 'hypernym', 'meronym', 'holonym', 'synonym', 'antonym']
        if intent_dict.get('semantic_relation') and intent_dict['semantic_relation'] not in valid_relations:
            warnings.append(f"Invalid semantic_relation: {intent_dict['semantic_relation']}")
        
        # Validate POS if present
        valid_pos = ['noun', 'verb', 'adjective', 'adverb']
        if intent_dict.get('pos') and intent_dict['pos'] not in valid_pos:
            warnings.append(f"Invalid pos: {intent_dict['pos']}")
        
        # Check consistency: semantic queries need lemma
        if intent_dict.get('query_type') == 'complit_semantic':
            if not intent_dict.get('lemma'):
                warnings.append("Semantic query requires lemma")
            if not intent_dict.get('semantic_relation'):
                warnings.append("Semantic query requires semantic_relation")
        
        # Check consistency: definition queries need pattern
        if intent_dict.get('query_type') == 'complit_definitions':
            if not intent_dict.get('definition_pattern'):
                warnings.append("Definition query requires definition_pattern")
        
        return warnings
    
    def _create_fallback_intent(self, user_query: str, error: str) -> Dict:
        """Create a safe fallback intent when LLM fails."""
        return {
            'query_type': 'basic_lemma_lookup',
            'required_resources': ['liita_lemma_bank'],
            'lemma': None,
            'pos': None,
            'definition_pattern': None,
            'pattern_type': None,
            'written_form_pattern': None,
            'semantic_relation': None,
            'filters': [],
            'aggregation': None,
            'retrieve_definitions': False,
            'retrieve_examples': False,
            'include_italian_written_rep': True,
            'complexity_score': 1,
            'user_query': user_query,
            'reasoning': f'Fallback intent due to error: {error}'
        }


# ============================================================================
# Mock LLM Client for Testing
# ============================================================================

class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""

    def complete(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Mock completion that returns a predefined response based on the query.
        In production, this would call the actual LLM API (use create_llm_client from llm.py).
        """
        
        # Extract query from prompt
        if 'How many nouns' in prompt:
            return json.dumps({
                "query_type": "basic_lemma_lookup",
                "required_resources": ["liita_lemma_bank"],
                "lemma": None,
                "pos": "noun",
                "definition_pattern": None,
                "pattern_type": None,
                "written_form_pattern": None,
                "semantic_relation": None,
                "filters": [],
                "aggregation": {
                    "type": "count",
                    "aggregate_var": None,
                    "group_by_vars": [],
                    "separator": None,
                    "order_by": None
                },
                "retrieve_definitions": False,
                "retrieve_examples": False,
                "include_italian_written_rep": False,
                "complexity_score": 1,
                "user_query": "How many nouns are in LiITA?",
                "reasoning": "Simple count aggregation on LiITA lemmas"
            })
        
        elif 'hyponyms of' in prompt.lower() and 'colore' in prompt.lower():
            return json.dumps({
                "query_type": "complit_semantic",
                "required_resources": ["complit", "liita_lemma_bank", "parmigiano"],
                "lemma": "colore",
                "pos": "noun",
                "definition_pattern": None,
                "pattern_type": None,
                "written_form_pattern": None,
                "semantic_relation": "hyponym",
                "filters": [],
                "aggregation": {
                    "type": "group_concat",
                    "aggregate_var": "definition",
                    "group_by_vars": ["relatedSense", "liitaLemma", "parmigianoLemma", "parmigianoWR"],
                    "separator": "; ",
                    "order_by": {"var": "parmigianoWR", "direction": "ASC"}
                },
                "retrieve_definitions": True,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 5,
                "user_query": prompt.split('Query: "')[1].split('"')[0] if 'Query: "' in prompt else "hyponyms query",
                "reasoning": "Complex semantic query with dialect translation"
            })
        
        # Default fallback
        return json.dumps({
            "query_type": "basic_lemma_lookup",
            "required_resources": ["liita_lemma_bank"],
            "lemma": None,
            "pos": None,
            "definition_pattern": None,
            "pattern_type": None,
            "written_form_pattern": None,
            "semantic_relation": None,
            "filters": [],
            "aggregation": None,
            "retrieve_definitions": False,
            "retrieve_examples": False,
            "include_italian_written_rep": True,
            "complexity_score": 1,
            "user_query": "unknown query",
            "reasoning": "Generic fallback"
        })


# ============================================================================
# Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LiITA INTENT ANALYZER - DEMONSTRATION")
    print("=" * 70)
    
    # Create analyzer with mock client
    mock_client = MockLLMClient()
    analyzer = IntentAnalyzer(mock_client)
    
    # Test queries
    test_queries = [
        "How many nouns are in LiITA?",
        "Find hyponyms of 'colore' with Parmigiano translations",
        "What are Italian words whose definition starts with 'uccello'?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {query}")
        print('=' * 70)
        
        # Analyze query
        intent_dict, warnings = analyzer.analyze(query, include_examples=3)
        
        # Display results
        print("\nExtracted Intent:")
        print(json.dumps(intent_dict, indent=2))
        
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n✓ No warnings - intent is valid")
        
        # Show key classifications
        print("\nKey Classifications:")
        print(f"  • Query Type: {intent_dict.get('query_type')}")
        print(f"  • Resources: {intent_dict.get('required_resources')}")
        print(f"  • Complexity: {intent_dict.get('complexity_score')}/5")
        if intent_dict.get('lemma'):
            print(f"  • Target Lemma: {intent_dict.get('lemma')}")
        if intent_dict.get('semantic_relation'):
            print(f"  • Relation: {intent_dict.get('semantic_relation')}")
        print(f"\n  Reasoning: {intent_dict.get('reasoning')}")
    
