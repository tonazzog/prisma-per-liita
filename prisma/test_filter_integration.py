"""
PRISMA v2 Filter Integration Test
=================================

End-to-end integration tests for the LLM-driven filter system.
Tests the full pipeline: LLM output -> IntentConverter -> Orchestrator -> SPARQL

This verifies that:
1. Filters in the new format are correctly processed
2. Legacy field extraction works for backwards compatibility
3. Pattern tools receive and apply filters correctly
4. Generated SPARQL contains the expected filter clauses
"""

import json
from typing import Dict, List, Optional, Tuple

from prisma_v2.translator import IntentConverter, Translator, TranslationResult
from prisma_v2.intent_analyzer import IntentAnalyzer
from prisma_v2.orchestrator import PatternOrchestrator, Intent
from prisma_v2.pattern_tools import PatternToolRegistry
from prisma_v2.assembler import PatternAssembler


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# ============================================================================
# Mock LLM Client for Integration Testing
# ============================================================================

class FilterTestMockLLM:
    """
    Mock LLM that returns intent dicts with the new filters format.
    Used to test the full pipeline without actual LLM calls.
    """

    def __init__(self):
        self.test_responses = {
            "masculine_nouns_a": {
                "query_type": "basic_lemma_lookup",
                "required_resources": ["liita_lemma_bank"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "pos", "value": "noun"},
                    {"property": "gender", "value": "masculine"},
                    {"pattern": "a$", "pattern_type": "ends_with"}
                ],
                "aggregation": None,
                "retrieve_definitions": False,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 2,
                "user_query": "Find all masculine nouns ending with 'a'",
                "reasoning": "LiITA lemma search with POS, gender, and pattern filters"
            },
            "positive_words": {
                "query_type": "sentix_polarity",
                "required_resources": ["liita_lemma_bank", "sentix"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "polarity", "value": "Positive"},
                    {"property": "polarity_value", "min_value": 0.5}
                ],
                "aggregation": None,
                "retrieve_definitions": False,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 2,
                "user_query": "Find strongly positive words",
                "reasoning": "Sentix query with polarity type and value range filter"
            },
            "joy_words": {
                "query_type": "elita_emotion",
                "required_resources": ["liita_lemma_bank", "elita"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "emotion", "value": "Gioia"}
                ],
                "aggregation": None,
                "retrieve_definitions": False,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 2,
                "user_query": "Find words associated with joy",
                "reasoning": "ELIta query filtering by Gioia emotion"
            },
            "sicilian_nouns": {
                "query_type": "dialect_pattern_search",
                "required_resources": ["sicilian", "liita_lemma_bank"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "pos", "value": "noun"},
                    {"pattern": "ìa$", "pattern_type": "ends_with"}
                ],
                "aggregation": None,
                "retrieve_definitions": False,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 3,
                "user_query": "Find Sicilian nouns ending in 'ìa'",
                "reasoning": "Sicilian dialect search with POS and pattern filter"
            },
            "definition_search": {
                "query_type": "complit_definitions",
                "required_resources": ["complit"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "pos", "value": "noun"},
                    {"pattern": "animale", "pattern_type": "contains", "target": "definition"}
                ],
                "aggregation": None,
                "retrieve_definitions": True,
                "retrieve_examples": False,
                "include_italian_written_rep": True,
                "complexity_score": 2,
                "user_query": "Find nouns whose definition contains 'animale'",
                "reasoning": "CompL-it definition search with POS and pattern filter"
            },
            "count_verbs": {
                "query_type": "basic_lemma_lookup",
                "required_resources": ["liita_lemma_bank"],
                "lemma": None,
                "semantic_relation": None,
                "filters": [
                    {"property": "pos", "value": "verb"}
                ],
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
                "user_query": "Count all verbs in LiITA",
                "reasoning": "Count aggregation with POS filter"
            }
        }
        self.current_test = None

    def set_test(self, test_name: str):
        """Set which test response to return"""
        self.current_test = test_name

    def complete(self, prompt: str, system: Optional[str] = None,
                 temperature: Optional[float] = None, **kwargs) -> str:
        """Return the appropriate test response as JSON"""
        if self.current_test and self.current_test in self.test_responses:
            return json.dumps(self.test_responses[self.current_test])
        return json.dumps(self.test_responses["masculine_nouns_a"])


# ============================================================================
# Integration Tests
# ============================================================================

def test_basic_filter_integration():
    """Test basic LiITA query with POS, gender, and pattern filters"""
    print_section("TEST: Basic Filter Integration (Masculine Nouns ending with 'a')")

    # Setup
    mock_llm = FilterTestMockLLM()
    mock_llm.set_test("masculine_nouns_a")
    translator = Translator(mock_llm, verbose=False)

    # Translate
    result = translator.translate("Find all masculine nouns ending with 'a'")

    # Verify success
    print(f"Success: {result.success}")
    assert result.success, f"Translation failed: {result.error_message}"

    # Verify intent extraction
    print(f"\nIntent extracted:")
    print(f"  Query type: {result.intent_dict.get('query_type')}")
    print(f"  Filters: {result.intent_dict.get('filters')}")

    # Verify SPARQL output
    print(f"\nGenerated SPARQL:\n{result.sparql_query}")

    # Check SPARQL contains expected elements
    sparql = result.sparql_query.lower()
    checks = [
        ("lila:haspos" in sparql or "haspos" in sparql, "POS filter"),
        ("lila:hasgender" in sparql or "hasgender" in sparql, "Gender filter"),
        ("strends" in sparql or "regex" in sparql, "Pattern filter"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name} present in SPARQL")
        if not check:
            all_passed = False

    return all_passed


def test_sentix_filter_integration():
    """Test Sentix polarity query with value range filter"""
    print_section("TEST: Sentix Filter Integration (Positive words > 0.5)")

    # Setup
    mock_llm = FilterTestMockLLM()
    mock_llm.set_test("positive_words")
    translator = Translator(mock_llm, verbose=False)

    # Translate
    result = translator.translate("Find strongly positive words")

    # Verify success
    print(f"Success: {result.success}")
    assert result.success, f"Translation failed: {result.error_message}"

    # Verify SPARQL output
    print(f"\nGenerated SPARQL:\n{result.sparql_query}")

    # Check SPARQL contains expected elements
    sparql = result.sparql_query
    checks = [
        ("marl:hasPolarity" in sparql or "hasPolarity" in sparql, "Polarity filter"),
        (">=" in sparql or "0.5" in sparql, "Polarity value range"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name} present in SPARQL")
        if not check:
            all_passed = False

    return all_passed


def test_elita_filter_integration():
    """Test ELIta emotion query with emotion filter"""
    print_section("TEST: ELIta Filter Integration (Joy words)")

    # Setup
    mock_llm = FilterTestMockLLM()
    mock_llm.set_test("joy_words")
    translator = Translator(mock_llm, verbose=False)

    # Translate
    result = translator.translate("Find words associated with joy")

    # Verify success
    print(f"Success: {result.success}")
    assert result.success, f"Translation failed: {result.error_message}"

    # Verify SPARQL output
    print(f"\nGenerated SPARQL:\n{result.sparql_query}")

    # Check SPARQL contains expected elements
    sparql = result.sparql_query
    checks = [
        ("elita:HasEmotion" in sparql or "HasEmotion" in sparql, "Emotion property"),
        ("elita:Gioia" in sparql or "Gioia" in sparql, "Joy emotion value"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name} present in SPARQL")
        if not check:
            all_passed = False

    return all_passed


def test_dialect_filter_integration():
    """Test Sicilian dialect query with pattern filter"""
    print_section("TEST: Dialect Filter Integration (Sicilian nouns ending in 'ìa')")

    # Setup
    mock_llm = FilterTestMockLLM()
    mock_llm.set_test("sicilian_nouns")
    translator = Translator(mock_llm, verbose=False)

    # Translate
    result = translator.translate("Find Sicilian nouns ending in 'ìa'")

    # Verify success
    print(f"Success: {result.success}")
    assert result.success, f"Translation failed: {result.error_message}"

    # Verify SPARQL output
    print(f"\nGenerated SPARQL:\n{result.sparql_query}")

    # Check SPARQL contains expected elements
    sparql = result.sparql_query
    checks = [
        ("Sicilian" in sparql or "sicilian" in sparql.lower(), "Sicilian graph reference"),
        ("strends" in sparql.lower() or "ìa" in sparql, "Pattern filter for 'ìa'"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name} present in SPARQL")
        if not check:
            all_passed = False

    return all_passed


def test_aggregation_with_filter():
    """Test count aggregation with POS filter"""
    print_section("TEST: Aggregation with Filter (Count verbs)")

    # Setup
    mock_llm = FilterTestMockLLM()
    mock_llm.set_test("count_verbs")
    translator = Translator(mock_llm, verbose=False)

    # Translate
    result = translator.translate("Count all verbs in LiITA")

    # Verify success
    print(f"Success: {result.success}")
    assert result.success, f"Translation failed: {result.error_message}"

    # Verify SPARQL output
    print(f"\nGenerated SPARQL:\n{result.sparql_query}")

    # Check SPARQL contains expected elements
    sparql = result.sparql_query.lower()
    checks = [
        ("count(" in sparql, "COUNT aggregation"),
        ("lila:haspos" in sparql or "haspos" in sparql, "POS filter"),
        ("lila:verb" in sparql or "verb" in sparql, "Verb value"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name} present in SPARQL")
        if not check:
            all_passed = False

    return all_passed


def test_filter_validation_warnings():
    """Test that invalid filters generate appropriate warnings"""
    print_section("TEST: Filter Validation Warnings")

    # Setup
    converter = IntentConverter()

    # Test with invalid POS value
    intent_dict = {
        "query_type": "basic_lemma_lookup",
        "required_resources": ["liita_lemma_bank"],
        "filters": [
            {"property": "pos", "value": "invalid_pos_value"}
        ],
        "user_query": "Test invalid filter"
    }

    intent, warnings = converter.dict_to_intent(intent_dict)

    print(f"Warnings generated: {len(warnings)}")
    for w in warnings:
        print(f"  - {w}")

    # Check that we got a warning about invalid POS value
    has_warning = any("may not be valid" in w for w in warnings)
    print(f"\n{'[PASS]' if has_warning else '[FAIL]'} Invalid value warning generated")

    return has_warning


def test_legacy_field_extraction():
    """Test that legacy fields are correctly extracted from filters"""
    print_section("TEST: Legacy Field Extraction")

    converter = IntentConverter()

    # Intent with filters only (no legacy pos/gender fields)
    intent_dict = {
        "query_type": "basic_lemma_lookup",
        "required_resources": ["liita_lemma_bank"],
        "filters": [
            {"property": "pos", "value": "adjective"},
            {"property": "gender", "value": "feminine"},
            {"pattern": "^bel", "pattern_type": "starts_with"}
        ],
        "user_query": "Find feminine adjectives starting with 'bel'"
    }

    intent, warnings = converter.dict_to_intent(intent_dict)

    print(f"Extracted legacy fields:")
    print(f"  pos: {intent.pos}")
    print(f"  gender: {intent.gender}")
    print(f"  written_form_pattern: {intent.written_form_pattern}")
    print(f"  pattern_type: {intent.pattern_type}")

    checks = [
        (intent.pos == "adjective", "POS extracted correctly"),
        (intent.gender == "feminine", "Gender extracted correctly"),
        (intent.written_form_pattern == "^bel", "Pattern extracted correctly"),
        (intent.pattern_type == "starts_with", "Pattern type extracted correctly"),
    ]

    all_passed = True
    for check, name in checks:
        status = "[PASS]" if check else "[FAIL]"
        print(f"  {status} {name}")
        if not check:
            all_passed = False

    return all_passed


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("PRISMA v2 FILTER INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Filter Integration", test_basic_filter_integration),
        ("Sentix Filter Integration", test_sentix_filter_integration),
        ("ELIta Filter Integration", test_elita_filter_integration),
        ("Dialect Filter Integration", test_dialect_filter_integration),
        ("Aggregation with Filter", test_aggregation_with_filter),
        ("Filter Validation Warnings", test_filter_validation_warnings),
        ("Legacy Field Extraction", test_legacy_field_extraction),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            results.append((name, f"ERROR: {str(e)}"))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)

    for name, result in results:
        status = "[OK]" if result == "PASS" else "[FAIL]"
        print(f"{status} {name}: {result}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nFilter integration is working correctly!")
        print("The LLM-driven filter system is ready for production use.")
    else:
        print("\nSome tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
