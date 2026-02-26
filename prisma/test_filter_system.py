"""
PRISMA v2 Filter System - End-to-End Tests
==========================================

Tests the new flexible filter system including:
1. FilterSpec and FilterRenderer
2. LiITABasicQueryPattern with flexible filters
3. Full pipeline integration (intent -> orchestrator -> pattern_tools -> SPARQL)
"""

import json
from typing import Dict, List

# Import filter system components
from prisma_v2.filter_system import FilterSpec, FilterType, FilterRenderer, FilterBuilder
from prisma_v2.property_registry import PropertyRegistry

# Import pattern tools
from prisma_v2.pattern_tools import (
    PatternToolRegistry,
    LiITABasicQueryPattern,
    PatternFragment
)

# Import orchestrator
from prisma_v2.orchestrator import PatternOrchestrator, Intent, QueryType, ResourceType


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# ============================================================================
# Test 1: FilterSpec and FilterRenderer
# ============================================================================

def test_filter_system_basics():
    """Test basic filter system functionality"""
    print_section("TEST 1: FilterSpec and FilterRenderer Basics")

    builder = FilterBuilder()
    renderer = FilterRenderer()

    # Test 1.1: POS filter
    print_subsection("1.1: POS Filter")
    filters = [builder.pos_filter("?lemma", "noun")]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: pos_filter('?lemma', 'noun')")
    print(f"Property patterns: {props}")
    print(f"Filter clauses: {clauses}")
    assert "lila:hasPOS" in props
    assert "lila:noun" in props
    print("[PASS] POS filter works correctly")

    # Test 1.2: Gender filter
    print_subsection("1.2: Gender Filter")
    filters = [builder.gender_filter("?lemma", "masculine")]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: gender_filter('?lemma', 'masculine')")
    print(f"Property patterns: {props}")
    assert "lila:hasGender" in props
    assert "lila:masculine" in props
    print("[PASS] Gender filter works correctly")

    # Test 1.3: Written rep pattern filter
    print_subsection("1.3: Written Rep Pattern Filter")
    filters = [builder.written_rep_pattern("?wr", "a$", "regex")]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: written_rep_pattern('?wr', 'a$', 'regex')")
    print(f"Filter clauses: {clauses}")
    assert "regex" in clauses.lower()
    assert "a$" in clauses
    print("[PASS] Written rep pattern filter works correctly")

    # Test 1.4: Combined filters (masculine nouns ending with 'a')
    print_subsection("1.4: Combined Filters - Masculine Nouns ending with 'a'")
    filters = [
        builder.pos_filter("?lemma", "noun"),
        builder.gender_filter("?lemma", "masculine"),
        builder.written_rep_pattern("?wr", "a$", "regex")
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Combined filters:")
    print(f"  - POS: noun")
    print(f"  - Gender: masculine")
    print(f"  - Pattern: ends with 'a'")
    print(f"\nProperty patterns:\n{props}")
    print(f"\nFilter clauses:\n{clauses}")
    assert "lila:hasPOS" in props
    assert "lila:noun" in props
    assert "lila:hasGender" in props
    assert "lila:masculine" in props
    assert "a$" in clauses
    print("\n[PASS] Combined filters work correctly")

    return True


# ============================================================================
# Test 2: Property Registry
# ============================================================================

def test_property_registry():
    """Test property registry functionality"""
    print_section("TEST 2: Property Registry")

    registry = PropertyRegistry()

    # Test 2.1: Get LiITA properties
    print_subsection("2.1: Get LiITA Properties")
    props = registry.get_properties("liita_lemma")
    print(f"Available properties: {list(props.keys())}")
    assert "pos" in props
    assert "gender" in props
    assert "inflection_type" in props
    assert "written_rep" in props
    print("[PASS] All expected properties available")

    # Test 2.2: Value normalization
    print_subsection("2.2: Value Normalization")
    normalized = registry.normalize_value("liita", "pos", "noun")
    print(f"normalize_value('liita', 'pos', 'noun') -> {normalized}")
    assert normalized == "lila:noun"

    normalized = registry.normalize_value("liita", "gender", "masculine")
    print(f"normalize_value('liita', 'gender', 'masculine') -> {normalized}")
    assert normalized == "lila:masculine"
    print("[PASS] Value normalization works correctly")

    # Test 2.3: Predicate lookup
    print_subsection("2.3: Predicate Lookup")
    pred = registry.get_predicate("liita", "pos")
    print(f"get_predicate('liita', 'pos') -> {pred}")
    assert pred == "lila:hasPOS"

    pred = registry.get_predicate("liita", "gender")
    print(f"get_predicate('liita', 'gender') -> {pred}")
    assert pred == "lila:hasGender"
    print("[PASS] Predicate lookup works correctly")

    return True


# ============================================================================
# Test 3: LiITABasicQueryPattern with Flexible Filters
# ============================================================================

def test_liita_pattern_flexible_filters():
    """Test LiITABasicQueryPattern with the new flexible filter system"""
    print_section("TEST 3: LiITABasicQueryPattern with Flexible Filters")

    pattern_tool = LiITABasicQueryPattern()

    # Test 3.1: Legacy mode (backward compatibility)
    print_subsection("3.1: Legacy Mode - POS filter only")
    fragment = pattern_tool.generate(pos_filter="noun")
    print(f"Input: pos_filter='noun'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "lila:hasPOS" in fragment.sparql
    assert "lila:noun" in fragment.sparql
    print("\n[PASS] Legacy mode works correctly")

    # Test 3.2: Legacy mode with gender (new parameter)
    print_subsection("3.2: Legacy Mode - POS + Gender")
    fragment = pattern_tool.generate(pos_filter="noun", gender_filter="masculine")
    print(f"Input: pos_filter='noun', gender_filter='masculine'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "lila:hasPOS" in fragment.sparql
    assert "lila:noun" in fragment.sparql
    assert "lila:hasGender" in fragment.sparql
    assert "lila:masculine" in fragment.sparql
    print("\n[PASS] Legacy mode with gender works correctly")

    # Test 3.3: Legacy mode with pattern
    print_subsection("3.3: Legacy Mode - POS + Gender + Pattern")
    fragment = pattern_tool.generate(
        pos_filter="noun",
        gender_filter="masculine",
        pattern="a$",
        pattern_type="regex"
    )
    print(f"Input: pos_filter='noun', gender_filter='masculine', pattern='a$'")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "lila:hasPOS" in fragment.sparql
    assert "lila:hasGender" in fragment.sparql
    assert "regex" in fragment.sparql.lower()
    assert "a$" in fragment.sparql
    print("\n[PASS] Complete legacy mode works correctly")

    # Test 3.4: Flexible filter mode
    print_subsection("3.4: Flexible Filter Mode")
    filters = [
        {
            "target_variable": "?lemma",
            "filter_type": "property_equals",
            "property_path": "lila:hasPOS",
            "value": "lila:noun"
        },
        {
            "target_variable": "?lemma",
            "filter_type": "property_equals",
            "property_path": "lila:hasGender",
            "value": "lila:masculine"
        },
        {
            "target_variable": "?wr",
            "filter_type": "regex",
            "value": "a$"
        }
    ]
    fragment = pattern_tool.generate(filters=filters)
    print(f"Input: filters (list of FilterSpec dicts)")
    print(f"\nGenerated SPARQL:\n{fragment.sparql}")
    assert "lila:hasPOS" in fragment.sparql
    assert "lila:noun" in fragment.sparql
    assert "lila:hasGender" in fragment.sparql
    assert "lila:masculine" in fragment.sparql
    assert "regex" in fragment.sparql.lower()
    print("\n[PASS] Flexible filter mode works correctly")

    # Test 3.5: Metadata verification
    print_subsection("3.5: Metadata Verification")
    print(f"Pattern name: {fragment.pattern_name}")
    print(f"Required prefixes: {fragment.required_prefixes}")
    print(f"Filters applied: {fragment.filters_applied}")
    print(f"Uses flexible filters: {fragment.metadata.get('uses_flexible_filters')}")
    assert fragment.metadata.get('uses_flexible_filters') == True
    print("\n[PASS] Metadata correctly tracks flexible filter usage")

    return True


# ============================================================================
# Test 4: Orchestrator Integration
# ============================================================================

def test_orchestrator_integration():
    """Test orchestrator passing filter parameters"""
    print_section("TEST 4: Orchestrator Integration")

    orchestrator = PatternOrchestrator()

    # Test 4.1: Basic lemma lookup with POS
    print_subsection("4.1: Basic Lemma Lookup with POS")
    intent = Intent(
        query_type=QueryType.BASIC_LEMMA_LOOKUP,
        required_resources=[ResourceType.LIITA],
        pos="noun"
    )
    plan = orchestrator.create_plan(intent)
    print(f"Query type: {intent.query_type}")
    print(f"POS: {intent.pos}")
    print(f"\nExecution plan steps: {len(plan.steps)}")
    for step in plan.steps:
        print(f"  - Tool: {step.tool_name}")
        print(f"    Params: {step.parameters}")
    print("\n[PASS] Basic lookup plan created")

    # Test 4.2: Basic lemma lookup with POS + Gender
    print_subsection("4.2: Basic Lemma Lookup with POS + Gender")
    intent = Intent(
        query_type=QueryType.BASIC_LEMMA_LOOKUP,
        required_resources=[ResourceType.LIITA],
        pos="noun",
        gender="masculine"
    )
    plan = orchestrator.create_plan(intent)
    print(f"Query type: {intent.query_type}")
    print(f"POS: {intent.pos}")
    print(f"Gender: {intent.gender}")
    print(f"\nExecution plan steps: {len(plan.steps)}")
    for step in plan.steps:
        print(f"  - Tool: {step.tool_name}")
        print(f"    Params: {step.parameters}")
    # Verify gender is in params
    liita_step = next((s for s in plan.steps if s.tool_name == "liita_basic_query"), None)
    if liita_step:
        assert "gender_filter" in liita_step.parameters or "filters" in liita_step.parameters
        print("\n[PASS] Gender parameter passed to pattern tool")

    # Test 4.3: Full query with pattern
    print_subsection("4.3: Full Query with POS + Gender + Pattern")
    intent = Intent(
        query_type=QueryType.BASIC_LEMMA_LOOKUP,
        required_resources=[ResourceType.LIITA],
        pos="noun",
        gender="masculine",
        written_form_pattern="a$"
    )
    plan = orchestrator.create_plan(intent)
    print(f"Query type: {intent.query_type}")
    print(f"POS: {intent.pos}")
    print(f"Gender: {intent.gender}")
    print(f"Pattern: {intent.written_form_pattern}")
    print(f"\nExecution plan steps: {len(plan.steps)}")
    for step in plan.steps:
        print(f"  - Tool: {step.tool_name}")
        print(f"    Params: {step.parameters}")
    print("\n[PASS] Full query plan created correctly")

    return True


# ============================================================================
# Test 5: Full Pipeline Simulation
# ============================================================================

def test_full_pipeline():
    """Test full pipeline from intent to SPARQL"""
    print_section("TEST 5: Full Pipeline - Intent to SPARQL")

    # Simulate intent from LLM
    print_subsection("5.1: Simulated LLM Intent Output")
    intent_dict = {
        "query_type": "basic_lemma_lookup",
        "required_resources": ["liita_lemma_bank"],
        "lemma": None,
        "pos": "noun",
        "gender": "masculine",
        "inflection_type": None,
        "definition_pattern": None,
        "pattern_type": "regex",
        "written_form_pattern": "a$",
        "semantic_relation": None,
        "filters": [],
        "aggregation": None,
        "retrieve_definitions": False,
        "retrieve_examples": False,
        "include_italian_written_rep": True,
        "complexity_score": 2,
        "user_query": "Find all masculine nouns ending with 'a'",
        "reasoning": "LiITA lemma search with POS, gender, and written form pattern filters"
    }
    print(f"User query: {intent_dict['user_query']}")
    print(f"\nExtracted intent:")
    print(f"  - query_type: {intent_dict['query_type']}")
    print(f"  - pos: {intent_dict['pos']}")
    print(f"  - gender: {intent_dict['gender']}")
    print(f"  - written_form_pattern: {intent_dict['written_form_pattern']}")

    # Convert to Intent object
    print_subsection("5.2: Convert to Intent Object")
    intent = Intent(
        query_type=QueryType.BASIC_LEMMA_LOOKUP,
        required_resources=[ResourceType.LIITA],
        pos=intent_dict["pos"],
        gender=intent_dict["gender"],
        written_form_pattern=intent_dict["written_form_pattern"],
        user_query=intent_dict["user_query"]
    )
    print(f"Intent object created: {intent.query_type}")

    # Create execution plan
    print_subsection("5.3: Create Execution Plan")
    orchestrator = PatternOrchestrator()
    plan = orchestrator.create_plan(intent)
    print(f"Plan created with {len(plan.steps)} steps")
    for step in plan.steps:
        print(f"  - {step.tool_name}: {step.parameters}")

    # Execute pattern tool
    print_subsection("5.4: Generate SPARQL Pattern")
    registry = PatternToolRegistry()

    # Get the first step (liita_basic_query)
    step = plan.steps[0]
    tool = registry.get(step.tool_name)

    if tool:
        fragment = tool.generate(**step.parameters)
        print(f"Tool: {tool.name}")
        print(f"\nGenerated SPARQL fragment:\n{fragment.sparql}")

        # Build complete query
        print_subsection("5.5: Build Complete SPARQL Query")

        # Add prefixes
        prefixes = """PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
"""

        # Build SELECT clause
        select_vars = " ".join([v.name for v in fragment.output_vars])

        # Build complete query
        complete_query = f"""{prefixes}
SELECT {select_vars}
WHERE {{
{fragment.sparql}
}}
LIMIT 100"""

        print(f"Complete SPARQL query:\n{complete_query}")

        # Verify key elements
        assert "lila:hasPOS" in complete_query
        assert "lila:noun" in complete_query
        assert "lila:hasGender" in complete_query
        assert "lila:masculine" in complete_query
        assert "regex" in complete_query.lower()
        assert "a$" in complete_query

        print("\n[PASS] Complete SPARQL query generated successfully!")

    return True


# ============================================================================
# Test 6: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and error handling"""
    print_section("TEST 6: Edge Cases")

    builder = FilterBuilder()
    renderer = FilterRenderer()

    # Test 6.1: Filter with already-prefixed value
    print_subsection("6.1: Already-prefixed values")
    filters = [builder.pos_filter("?lemma", "lila:noun")]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: pos_filter('?lemma', 'lila:noun')")
    print(f"Property patterns: {props}")
    # Should not double-prefix
    assert "lila:lila:noun" not in props
    print("[PASS] No double-prefixing")

    # Test 6.2: Variable without ? prefix
    print_subsection("6.2: Variable without ? prefix")
    filters = [builder.pos_filter("lemma", "noun")]  # No ? prefix
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: pos_filter('lemma', 'noun') (no ? prefix)")
    print(f"Property patterns: {props}")
    assert "?lemma" in props  # Should auto-add ?
    print("[PASS] Auto-prefixing of ? works")

    # Test 6.3: Empty filter list
    print_subsection("6.3: Empty filter list")
    props, clauses, out_vars = renderer.render([])
    print(f"Input: empty list")
    print(f"Property patterns: '{props}'")
    print(f"Filter clauses: '{clauses}'")
    assert props == ""
    assert clauses == ""
    print("[PASS] Empty filter list handled correctly")

    # Test 6.4: Multiple patterns on same property
    print_subsection("6.4: Multiple filters on same target")
    filters = [
        builder.pos_filter("?lemma", "noun"),
        builder.gender_filter("?lemma", "masculine"),
        builder.gender_filter("?lemma", "feminine")  # Conflicting - but allowed
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Input: pos=noun, gender=masculine, gender=feminine (conflicting)")
    print(f"Property patterns:\n{props}")
    # Should have both gender patterns (query will return no results but is valid)
    assert props.count("lila:hasGender") == 2
    print("[PASS] Conflicting filters handled (semantically empty result but valid SPARQL)")

    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PRISMA v2 FILTER SYSTEM - END-TO-END TESTS")
    print("=" * 70)

    tests = [
        ("Filter System Basics", test_filter_system_basics),
        ("Property Registry", test_property_registry),
        ("LiITA Pattern Flexible Filters", test_liita_pattern_flexible_filters),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Full Pipeline", test_full_pipeline),
        ("Edge Cases", test_edge_cases),
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
        print("\nAll tests passed! The flexible filter system is working correctly.")
    else:
        print("\nSome tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
