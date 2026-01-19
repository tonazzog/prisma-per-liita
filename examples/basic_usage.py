#!/usr/bin/env python3
"""
PRISMA per LiITA - Basic Usage Example
======================================

This script demonstrates how to use PRISMA to translate
natural language queries into SPARQL.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prisma import Translator, create_llm_client


def main():
    print("=" * 60)
    print("PRISMA per LiITA - Basic Usage Example")
    print("=" * 60)

    # ========================================================================
    # Step 1: Create an LLM client
    # ========================================================================
    print("\n[1] Creating LLM client...")

    # You can use any supported provider:
    # - "mistral" (default)
    # - "anthropic"
    # - "openai"
    # - "gemini"
    # - "ollama" (local, no API key needed)

    try:
        client = create_llm_client(
            provider="mistral",
            # api_key="your-api-key"  # Or set MISTRAL_API_KEY env var
        )
        print("  [OK] LLM client created")
    except ValueError as e:
        print(f"  [ERROR] {e}")
        print("\n  Please set your API key:")
        print("    export MISTRAL_API_KEY='your-key'")
        print("  Or use Ollama for local inference:")
        print("    client = create_llm_client(provider='ollama')")
        return

    # ========================================================================
    # Step 2: Create the translator
    # ========================================================================
    print("\n[2] Creating translator...")
    translator = Translator(client, verbose=True)
    print("  [OK] Translator ready")

    # ========================================================================
    # Step 3: Translate queries
    # ========================================================================
    print("\n[3] Translating queries...")

    queries = [
        "What is the polarity of 'amore'?",
        "Find the hyponyms of 'colore'",
        "How many nouns are in LiITA?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        result = translator.translate(query)

        if result.success:
            print(f"\n[OK] Translation successful!")
            print(f"  - Query type: {result.intent_dict.get('query_type')}")
            print(f"  - Resources: {result.intent_dict.get('required_resources')}")
            print(f"  - Steps: {result.total_steps}")
            print(f"  - Time: {result.processing_time_ms:.1f}ms")

            print(f"\nGenerated SPARQL:")
            print("-" * 40)
            print(result.sparql_query)
        else:
            print(f"\n[FAIL] Translation failed: {result.error_message}")

    # ========================================================================
    # Step 4: Show detailed explanation
    # ========================================================================
    print("\n" + "=" * 60)
    print("Detailed Explanation (last query)")
    print("=" * 60)

    explanation = translator.explain(result)
    print(explanation)


if __name__ == "__main__":
    main()
