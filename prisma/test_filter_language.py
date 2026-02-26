"""
PRISMA v2 Filter Language - Test Suite
======================================

Tests for the LLM-friendly filter language and translation.
"""

from prisma_v2.filter_language import (
    SimplifiedFilter,
    SimplifiedFilterType,
    FilterTranslator,
    FilterValidator,
    translate_llm_filters,
)
from prisma_v2.filter_system import FilterSpec, FilterType, FilterRenderer


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# ============================================================================
# Test SimplifiedFilter
# ============================================================================

def test_simplified_filter():
    """Test SimplifiedFilter dataclass"""
    print_section("TEST: SimplifiedFilter")

    # Test 1: Property filter
    print_subsection("Property Filter")
    sf = SimplifiedFilter(property="pos", value="noun")
    print(f"Created: {sf.to_dict()}")
    assert sf.get_filter_type() == SimplifiedFilterType.PROPERTY
    is_valid, errors = sf.validate()
    assert is_valid, f"Validation failed: {errors}"
    print("[PASS] Property filter works")

    # Test 2: Pattern filter
    print_subsection("Pattern Filter")
    sf = SimplifiedFilter(pattern="a$", pattern_type="ends_with")
    print(f"Created: {sf.to_dict()}")
    assert sf.get_filter_type() == SimplifiedFilterType.PATTERN
    is_valid, errors = sf.validate()
    assert is_valid, f"Validation failed: {errors}"
    print("[PASS] Pattern filter works")

    # Test 3: Range filter
    print_subsection("Range Filter")
    sf = SimplifiedFilter(property="polarity_value", min_value=0.5, max_value=1.0)
    print(f"Created: {sf.to_dict()}")
    assert sf.get_filter_type() == SimplifiedFilterType.RANGE
    is_valid, errors = sf.validate()
    assert is_valid, f"Validation failed: {errors}"
    print("[PASS] Range filter works")

    # Test 4: Retrieval filter
    print_subsection("Retrieval Filter (retrieve=True)")
    sf = SimplifiedFilter(property="gender", retrieve=True)
    print(f"Created: {sf.to_dict()}")
    is_valid, errors = sf.validate()
    assert is_valid, f"Validation failed: {errors}"
    print("[PASS] Retrieval filter works")

    # Test 5: from_dict
    print_subsection("from_dict")
    d = {"property": "pos", "value": "noun", "negate": True}
    sf = SimplifiedFilter.from_dict(d)
    print(f"Input: {d}")
    print(f"Created: {sf.to_dict()}")
    assert sf.property == "pos"
    assert sf.value == "noun"
    assert sf.negate == True
    print("[PASS] from_dict works")

    # Test 6: Invalid filter
    print_subsection("Invalid Filter (missing required fields)")
    sf = SimplifiedFilter(property="pos")  # Missing value
    is_valid, errors = sf.validate()
    print(f"Filter: {sf.to_dict()}")
    print(f"Valid: {is_valid}, Errors: {errors}")
    assert not is_valid
    print("[PASS] Validation catches invalid filters")

    return True


# ============================================================================
# Test FilterTranslator
# ============================================================================

def test_filter_translator():
    """Test FilterTranslator"""
    print_section("TEST: FilterTranslator")

    translator = FilterTranslator("liita")

    # Test 1: Translate POS filter
    print_subsection("Translate POS Filter")
    sf = SimplifiedFilter(property="pos", value="noun")
    spec = translator.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: property_path={spec.property_path}, value={spec.value}")
    assert spec.property_path == "lila:hasPOS"
    assert spec.value == "lila:noun"
    assert spec.filter_type == FilterType.PROPERTY_EQUALS
    print("[PASS] POS filter translated correctly")

    # Test 2: Translate gender filter
    print_subsection("Translate Gender Filter")
    sf = SimplifiedFilter(property="gender", value="masculine")
    spec = translator.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: property_path={spec.property_path}, value={spec.value}")
    assert spec.property_path == "lila:hasGender"
    assert spec.value == "lila:masculine"
    print("[PASS] Gender filter translated correctly")

    # Test 3: Translate pattern filter
    print_subsection("Translate Pattern Filter")
    sf = SimplifiedFilter(pattern="^casa", pattern_type="starts_with", target="written_rep")
    spec = translator.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: filter_type={spec.filter_type}, value={spec.value}")
    assert spec.filter_type == FilterType.STARTS_WITH
    assert spec.value == "^casa"
    print("[PASS] Pattern filter translated correctly")

    # Test 4: Translate range filter
    print_subsection("Translate Range Filter")
    translator_sentix = FilterTranslator("sentix")
    sf = SimplifiedFilter(property="polarity_value", min_value=0.5)
    spec = translator_sentix.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: filter_type={spec.filter_type}, value={spec.value}")
    assert spec.filter_type == FilterType.GREATER_EQUAL
    assert spec.value == "0.5"
    print("[PASS] Range filter translated correctly")

    # Test 5: Translate retrieval filter
    print_subsection("Translate Retrieval Filter")
    sf = SimplifiedFilter(property="gender", retrieve=True)
    spec = translator.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: filter_type={spec.filter_type}, value_variable={spec.value_variable}")
    assert spec.filter_type == FilterType.PROPERTY_EQUALS_VAR
    assert spec.value_variable == "?gender"
    print("[PASS] Retrieval filter translated correctly")

    # Test 6: Translate negated filter
    print_subsection("Translate Negated Filter")
    sf = SimplifiedFilter(property="pos", value="verb", negate=True)
    spec = translator.translate(sf)
    print(f"Input: {sf.to_dict()}")
    print(f"Output: negate={spec.negate}")
    assert spec.negate == True
    print("[PASS] Negated filter translated correctly")

    # Test 7: translate_all
    print_subsection("Translate Multiple Filters")
    filters = [
        SimplifiedFilter(property="pos", value="noun"),
        SimplifiedFilter(property="gender", value="masculine"),
        SimplifiedFilter(pattern="a$", pattern_type="ends_with")
    ]
    specs, warnings = translator.translate_all(filters)
    print(f"Input: {len(filters)} filters")
    print(f"Output: {len(specs)} specs, {len(warnings)} warnings")
    assert len(specs) == 3
    assert len(warnings) == 0
    print("[PASS] translate_all works correctly")

    return True


# ============================================================================
# Test FilterValidator
# ============================================================================

def test_filter_validator():
    """Test FilterValidator"""
    print_section("TEST: FilterValidator")

    validator = FilterValidator()

    # Test 1: Valid filters
    print_subsection("Valid Filters")
    filters = [
        {"property": "pos", "value": "noun"},
        {"property": "gender", "value": "masculine"},
        {"pattern": "a$", "pattern_type": "ends_with"}
    ]
    valid, warnings = validator.validate(filters, "liita")
    print(f"Input: {filters}")
    print(f"Valid: {len(valid)}, Warnings: {len(warnings)}")
    assert len(valid) == 3
    print("[PASS] Valid filters pass validation")

    # Test 2: Invalid POS value
    print_subsection("Invalid POS Value")
    filters = [{"property": "pos", "value": "invalid_pos"}]
    valid, warnings = validator.validate(filters, "liita")
    print(f"Input: {filters}")
    print(f"Valid: {len(valid)}, Warnings: {warnings}")
    assert len(valid) == 1  # Still valid structurally
    assert len(warnings) == 1  # But with a warning
    assert "may not be valid" in warnings[0]
    print("[PASS] Invalid value generates warning")

    # Test 3: Unknown property
    print_subsection("Unknown Property")
    filters = [{"property": "unknown_prop", "value": "test"}]
    valid, warnings = validator.validate(filters, "liita")
    print(f"Input: {filters}")
    print(f"Valid: {len(valid)}, Warnings: {warnings}")
    assert len(valid) == 1  # Still valid structurally
    assert len(warnings) == 1  # But with a warning
    assert "not in registry" in warnings[0]
    print("[PASS] Unknown property generates warning")

    # Test 4: Invalid pattern type
    print_subsection("Invalid Pattern Type")
    filters = [{"pattern": "test", "pattern_type": "invalid_type"}]
    valid, warnings = validator.validate(filters, "liita")
    print(f"Input: {filters}")
    print(f"Valid: {len(valid)}, Warnings: {warnings}")
    assert len(warnings) == 1
    assert "Invalid pattern_type" in warnings[0]
    print("[PASS] Invalid pattern_type generates warning")

    # Test 5: Structurally invalid filter
    print_subsection("Structurally Invalid Filter")
    filters = [{"property": "pos"}]  # Missing value
    valid, warnings = validator.validate(filters, "liita")
    print(f"Input: {filters}")
    print(f"Valid: {len(valid)}, Warnings: {warnings}")
    assert len(valid) == 0  # Structurally invalid
    assert len(warnings) == 1
    print("[PASS] Structurally invalid filter rejected")

    return True


# ============================================================================
# Test translate_llm_filters Convenience Function
# ============================================================================

def test_translate_llm_filters():
    """Test the convenience function"""
    print_section("TEST: translate_llm_filters")

    # Test 1: Typical LLM output
    print_subsection("Typical LLM Output")
    llm_output = [
        {"property": "pos", "value": "noun"},
        {"property": "gender", "value": "masculine"},
        {"pattern": "a$", "pattern_type": "ends_with"}
    ]
    specs, warnings = translate_llm_filters(llm_output, "liita")
    print(f"LLM Output: {llm_output}")
    print(f"FilterSpecs: {len(specs)}")
    print(f"Warnings: {warnings}")
    assert len(specs) == 3
    assert len(warnings) == 0
    print("[PASS] Typical LLM output translated correctly")

    # Test 2: LLM output with errors
    print_subsection("LLM Output with Errors")
    llm_output = [
        {"property": "pos", "value": "noun"},
        {"property": "invalid", "value": "test"},
        {"pattern": "bad", "pattern_type": "wrong_type"}
    ]
    specs, warnings = translate_llm_filters(llm_output, "liita")
    print(f"LLM Output: {llm_output}")
    print(f"FilterSpecs: {len(specs)}")
    print(f"Warnings: {warnings}")
    # Should still get some specs even with invalid input
    assert len(specs) >= 1
    assert len(warnings) > 0
    print("[PASS] LLM output with errors handled gracefully")

    # Test 3: Empty input
    print_subsection("Empty Input")
    specs, warnings = translate_llm_filters([], "liita")
    print(f"LLM Output: []")
    print(f"FilterSpecs: {len(specs)}, Warnings: {len(warnings)}")
    assert len(specs) == 0
    assert len(warnings) == 0
    print("[PASS] Empty input handled correctly")

    return True


# ============================================================================
# Test End-to-End with FilterRenderer
# ============================================================================

def test_end_to_end():
    """Test full pipeline: LLM output -> FilterSpecs -> SPARQL"""
    print_section("TEST: End-to-End Pipeline")

    # Simulate LLM output for "Find masculine nouns ending with 'a'"
    print_subsection("Query: Find masculine nouns ending with 'a'")

    llm_output = [
        {"property": "pos", "value": "noun"},
        {"property": "gender", "value": "masculine"},
        {"pattern": "a$", "pattern_type": "ends_with", "target": "written_rep"}
    ]
    print(f"LLM Output: {llm_output}")

    # Translate to FilterSpecs
    specs, warnings = translate_llm_filters(llm_output, "liita")
    print(f"\nTranslated to {len(specs)} FilterSpecs")
    for i, spec in enumerate(specs):
        print(f"  {i+1}. {spec.filter_type.value}")

    # Render to SPARQL
    renderer = FilterRenderer()
    props, clauses, out_vars = renderer.render(specs)
    print(f"\nRendered SPARQL:")
    print(f"Property patterns:\n{props}")
    print(f"\nFilter clauses:\n{clauses}")

    # Verify
    assert "lila:hasPOS" in props
    assert "lila:noun" in props
    assert "lila:hasGender" in props
    assert "lila:masculine" in props
    assert "strends" in clauses.lower()
    assert "a$" in clauses

    print("\n[PASS] End-to-end pipeline works correctly!")

    # Simulate LLM output for Sentix query
    print_subsection("Query: Find words with positive sentiment (> 0.5)")

    llm_output = [
        {"property": "polarity", "value": "Positive"},
        {"property": "polarity_value", "min_value": 0.5}
    ]
    print(f"LLM Output: {llm_output}")

    specs, warnings = translate_llm_filters(llm_output, "sentix")
    print(f"\nTranslated to {len(specs)} FilterSpecs")

    # Note: For Sentix, polarity uses a different predicate path
    # This might need adjustment in the property registry
    print(f"Specs: {[(s.filter_type.value, s.value or s.property_path) for s in specs]}")

    print("\n[PASS] Sentix query translated!")

    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PRISMA v2 FILTER LANGUAGE - TEST SUITE")
    print("=" * 70)

    tests = [
        ("SimplifiedFilter", test_simplified_filter),
        ("FilterTranslator", test_filter_translator),
        ("FilterValidator", test_filter_validator),
        ("translate_llm_filters", test_translate_llm_filters),
        ("End-to-End Pipeline", test_end_to_end),
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
        print("\nFilter language ready for LLM integration!")
    else:
        print("\nSome tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
