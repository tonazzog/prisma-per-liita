"""
PRISMA per LiITA
================

Pattern-based Rules for Intent-driven SPARQL with Multiple-resource Assembly.
A transparent NL2SPARQL system for the LiITA (Linking Italian) knowledge base.

Features:
- LLM-based intent analysis (classification only, not generation)
- Deterministic pattern orchestration
- Template-based SPARQL assembly
- Full pipeline transparency

Usage:
    from prisma import Translator, create_llm_client

    # Create an LLM client
    client = create_llm_client(
        provider="mistral",  # or "anthropic", "openai", "gemini", "ollama"
        api_key="your-api-key"
    )

    # Create the translator
    translator = Translator(client)

    # Translate natural language to SPARQL
    result = translator.translate("What is the polarity of 'amore'?")
    print(result.sparql_query)

Author: PRISMA per LiITA Project
"""

__version__ = "0.1.0"
__author__ = "PRISMA per LiITA Project"

from .llm import (
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

__all__ = [
    # Version
    "__version__",
    # LLM
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
]
