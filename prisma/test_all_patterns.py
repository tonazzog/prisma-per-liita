"""
PRISMA v2 - Test All Pattern Tools with Flexible Filters
========================================================

Tests all refactored pattern tools to verify flexible filter support.
"""

from prisma_v2.filter_system import FilterSpec, FilterType, FilterRenderer, FilterBuilder
from prisma_v2.pattern_tools import (
    PatternToolRegistry,
    CompLitDefinitionSearchPattern,
    CompLitSemanticRelationPattern,
    SicilianPatternSearchPattern,
    ParmigianoPatternSearchPattern,
    SentixLinkingPattern,
    ELItaLinkingPattern,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def test_complit_definition_search():
    """Test CompLitDefinitionSearchPattern with flexible filters"""
    print_section("TEST: CompLitDefinitionSearchPattern")

    pattern = CompLitDefinitionSearchPattern()

    # Test 1: Legacy mode
    print_subsection("Legacy Mode")
    fragment = pattern.generate(
        definition_pattern="animale",
        pattern_type="contains",
        pos_filter="noun"
    )
    print(f"Definition pattern: 'animale', type: contains, POS: noun")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "contains" in fragment.sparql.lower()
    assert "animale" in fragment.sparql
    print("\n[PASS] Legacy mode works")

    # Test 2: Flexible filter mode
    print_subsection("Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?definition",
            "filter_type": "starts_with",
            "value": "tipo di"
        }
    ]
    fragment = pattern.generate(filters=filters)
    print(f"Filters: definition starts_with 'tipo di'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filter mode works")

    return True


def test_complit_semantic_relation():
    """Test CompLitSemanticRelationPattern with flexible filters"""
    print_section("TEST: CompLitSemanticRelationPattern")

    pattern = CompLitSemanticRelationPattern()

    # Test 1: Legacy mode - find hyponyms
    print_subsection("Legacy Mode - Find hyponyms")
    fragment = pattern.generate(
        lemma="colore",
        relation_type="hyponym",
        pos="noun"
    )
    print(f"Lemma: colore, Relation: hyponym, POS: noun")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "lexinfo:hypernym" in fragment.sparql  # hyponym query uses hypernym property
    assert "colore" in fragment.sparql
    print("\n[PASS] Legacy mode works")

    # Test 2: With additional flexible filters
    print_subsection("With Additional Flexible Filters")
    filters = [
        {
            "target_variable": "?definition",
            "filter_type": "contains",
            "value": "luce"
        }
    ]
    fragment = pattern.generate(
        lemma="colore",
        relation_type="hyponym",
        pos="noun",
        filters=filters
    )
    print(f"Added filter: definition contains 'luce'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filters added")

    return True


def test_sicilian_pattern_search():
    """Test SicilianPatternSearchPattern with flexible filters"""
    print_section("TEST: SicilianPatternSearchPattern")

    pattern = SicilianPatternSearchPattern()

    # Test 1: Legacy mode with regex
    print_subsection("Legacy Mode - Regex Pattern")
    fragment = pattern.generate(
        pattern="ia$",
        pos="noun",
        link_to_italian=True
    )
    print(f"Pattern: 'ia$' (regex), POS: noun")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "regex" in fragment.sparql.lower()
    assert "ia$" in fragment.sparql
    print("\n[PASS] Legacy regex mode works")

    # Test 2: Legacy mode with different pattern type
    print_subsection("Legacy Mode - Ends With Pattern")
    fragment = pattern.generate(
        pattern="ìa",
        pattern_type="ends_with",
        link_to_italian=False
    )
    print(f"Pattern: 'ia' (ends_with)")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "strends" in fragment.sparql.lower()
    print("\n[PASS] Legacy ends_with mode works")

    # Test 3: Flexible filter mode
    print_subsection("Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?sicilianWR",
            "filter_type": "starts_with",
            "value": "ca"
        }
    ]
    fragment = pattern.generate(filters=filters, link_to_italian=True)
    print(f"Filters: starts_with 'ca'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filter mode works")

    return True


def test_parmigiano_pattern_search():
    """Test ParmigianoPatternSearchPattern with flexible filters"""
    print_section("TEST: ParmigianoPatternSearchPattern")

    pattern = ParmigianoPatternSearchPattern()

    # Test 1: Legacy mode
    print_subsection("Legacy Mode")
    fragment = pattern.generate(
        pattern="^ca",
        pattern_type="regex",
        pos="verb"
    )
    print(f"Pattern: '^ca' (regex), POS: verb")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "Parmigiano" in fragment.sparql
    assert "^ca" in fragment.sparql
    print("\n[PASS] Legacy mode works")

    # Test 2: Flexible filter mode
    print_subsection("Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?parmigianoWR",
            "filter_type": "contains",
            "value": "acqua"
        }
    ]
    fragment = pattern.generate(filters=filters)
    print(f"Filters: contains 'acqua'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filter mode works")

    return True


def test_sentix_linking():
    """Test SentixLinkingPattern with flexible filters"""
    print_section("TEST: SentixLinkingPattern")

    pattern = SentixLinkingPattern()

    # Test 1: Legacy mode - filter by polarity
    print_subsection("Legacy Mode - Polarity Filter")
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        polarity_filter="Positive",
        retrieve_polarity_value=True
    )
    print(f"Polarity filter: Positive")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "Positive" in fragment.sparql
    assert "marl:hasPolarity" in fragment.sparql
    print("\n[PASS] Legacy polarity filter works")

    # Test 2: Legacy mode - polarity value range
    print_subsection("Legacy Mode - Polarity Value Range")
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        polarity_value_min=0.5,
        polarity_value_max=1.0
    )
    print(f"Polarity value range: 0.5 to 1.0")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert ">= 0.5" in fragment.sparql
    assert "<= 1.0" in fragment.sparql
    print("\n[PASS] Legacy value range filter works")

    # Test 3: Flexible filter mode
    print_subsection("Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?polarityValue",
            "filter_type": "greater_equal",
            "value": "0"
        }
    ]
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        filters=filters
    )
    print(f"Filters: polarityValue >= 0")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filter mode works")

    return True


def test_elita_linking():
    """Test ELItaLinkingPattern with flexible filters"""
    print_section("TEST: ELItaLinkingPattern")

    pattern = ELItaLinkingPattern()

    # Test 1: Legacy mode - filter by emotions
    print_subsection("Legacy Mode - Emotion Filter")
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        emotion_filters=["joy", "sadness"]
    )
    print(f"Emotion filters: ['joy', 'sadness']")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "elita:Gioia" in fragment.sparql
    assert "elita:Tristezza" in fragment.sparql
    assert "VALUES ?emotion" in fragment.sparql
    print("\n[PASS] Legacy emotion filter works")

    # Test 2: No emotion filter
    print_subsection("No Emotion Filter")
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        retrieve_emotion_label=True
    )
    print(f"No emotion filters, retrieve all")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "VALUES" not in fragment.sparql  # No filter means no VALUES clause
    assert "elita:HasEmotion" in fragment.sparql
    print("\n[PASS] No filter mode works")

    # Test 3: Flexible filter mode
    print_subsection("Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?emotionLabel",
            "filter_type": "regex",
            "value": "^G"
        }
    ]
    fragment = pattern.generate(
        liita_lemma_var="?lemma",
        filters=filters
    )
    print(f"Filters: emotionLabel regex '^G'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert fragment.metadata.get("uses_flexible_filters") == True
    print("\n[PASS] Flexible filter mode works")

    return True


def test_pattern_registry():
    """Test that all patterns are registered correctly"""
    print_section("TEST: Pattern Tool Registry")

    registry = PatternToolRegistry()
    tools = registry.list_tools()

    print(f"Registered tools ({len(tools)}):")
    for tool_name in tools:
        tool = registry.get(tool_name)
        has_init = hasattr(tool, '_filter_renderer')
        status = "[v2]" if has_init else "[v1]"
        print(f"  {status} {tool_name}: {tool.description}")

    # Verify key patterns are registered
    assert "complit_definition_search" in tools
    assert "complit_semantic_relation" in tools
    assert "sicilian_pattern_search" in tools
    assert "parmigiano_pattern_search" in tools
    assert "sentix_linking" in tools
    assert "elita_linking" in tools
    assert "liita_basic_query" in tools

    print("\n[PASS] All patterns registered correctly")
    return True


def run_all_tests():
    """Run all pattern tests"""
    print("\n" + "=" * 70)
    print("PRISMA v2 - ALL PATTERN TOOLS TEST SUITE")
    print("=" * 70)

    tests = [
        ("CompLit Definition Search", test_complit_definition_search),
        ("CompLit Semantic Relation", test_complit_semantic_relation),
        ("Sicilian Pattern Search", test_sicilian_pattern_search),
        ("Parmigiano Pattern Search", test_parmigiano_pattern_search),
        ("Sentix Linking", test_sentix_linking),
        ("ELIta Linking", test_elita_linking),
        ("Pattern Registry", test_pattern_registry),
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
        print("\nAll pattern tools support flexible filters correctly!")
    else:
        print("\nSome tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
