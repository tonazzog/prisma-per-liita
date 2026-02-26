"""
PRISMA per LiITA v2 (Experimental)
==================================

Pattern-based Rules for Intent-driven SPARQL with Multiple-resource Assembly.
Version 2 with flexible filter system and skeleton-based query generation.

This is the experimental version with:
- Flexible filter specifications (FilterSpec)
- Skeleton-based pattern generation with slots
- Dynamic property access from ontology
- Foundation for LLM-driven refinement

Usage:
    from prisma_v2 import Translator
    from shared import create_llm_client

    # Create an LLM client
    client = create_llm_client(
        provider="mistral",  # or "anthropic", "openai", "gemini", "ollama"
        api_key="your-api-key"
    )

    # Create the translator
    translator = Translator(client)

    # Translate natural language to SPARQL
    result = translator.translate("Find all masculine nouns ending with 'a'")
    print(result.sparql_query)

Author: PRISMA per LiITA Project
"""

__version__ = "0.2.0-experimental"
__author__ = "PRISMA per LiITA Project"

# Import LLM from shared module
from shared import (
    BaseLLM,
    MistralLLM,
    AnthropicLLM,
    OpenAILLM,
    GeminiLLM,
    OllamaLLM,
    create_llm_client,
)

from .translator import (
    Translator,
    TranslationResult,
    IntentConverter,
)

from .intent_analyzer import (
    IntentAnalyzer,
    INTENT_ANALYZER_SYSTEM_PROMPT,
)

from .orchestrator import (
    PatternOrchestrator,
    Intent,
    QueryType,
    ResourceType,
    SemanticRelationType,
    ExecutionPlan,
    PatternStep,
)

from .pattern_tools import (
    PatternToolRegistry,
    PatternFragment,
)

from .assembler import (
    PatternAssembler,
    AssemblyResult,
)

# v2 Filter System (new in v2)
from .filter_system import (
    FilterSpec,
    FilterType,
    FilterRenderer,
    FilterBuilder,
)

from .property_registry import (
    PropertyRegistry,
    PropertyInfo,
    ValueType,
)

__all__ = [
    # Version
    "__version__",
    # LLM (from shared)
    "BaseLLM",
    "MistralLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "GeminiLLM",
    "OllamaLLM",
    "create_llm_client",
    # Main API
    "Translator",
    "TranslationResult",
    "IntentConverter",
    # Intent Analysis
    "IntentAnalyzer",
    "INTENT_ANALYZER_SYSTEM_PROMPT",
    # Orchestration
    "PatternOrchestrator",
    "Intent",
    "QueryType",
    "ResourceType",
    "SemanticRelationType",
    "ExecutionPlan",
    "PatternStep",
    # Pattern Tools
    "PatternToolRegistry",
    "PatternFragment",
    # Assembly
    "PatternAssembler",
    "AssemblyResult",
    # Filter System (v2)
    "FilterSpec",
    "FilterType",
    "FilterRenderer",
    "FilterBuilder",
    "PropertyRegistry",
    "PropertyInfo",
    "ValueType",
]
