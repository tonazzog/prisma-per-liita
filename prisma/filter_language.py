"""
PRISMA v2 Filter Language
=========================

LLM-friendly filter specification language and translation to FilterSpecs.

This module provides:
- SimplifiedFilter: A high-level, LLM-friendly filter format
- FilterTranslator: Converts simplified filters to FilterSpecs
- FilterValidator: Validates LLM-generated filters

The goal is to allow the LLM to express filters using natural property names
(like "pos", "gender") rather than requiring knowledge of exact SPARQL predicates
(like "lila:hasPOS", "lila:hasGender").

Example LLM output:
    {
        "filters": [
            {"property": "pos", "value": "noun"},
            {"property": "gender", "value": "masculine"},
            {"pattern": "a$", "pattern_type": "ends_with"}
        ]
    }

This gets translated to proper FilterSpecs that the pattern tools can use.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from .filter_system import FilterSpec, FilterType, FilterBuilder
from .property_registry import PropertyRegistry, PropertyInfo, ValueType


# ============================================================================
# Simplified Filter Schema (LLM-Friendly)
# ============================================================================

class SimplifiedFilterType(Enum):
    """Types of simplified filters the LLM can express"""
    PROPERTY = "property"      # Filter by a property value (pos, gender, etc.)
    PATTERN = "pattern"        # Filter by text pattern (regex, starts_with, etc.)
    RANGE = "range"            # Filter by numeric range (polarity_value, etc.)
    EXISTENCE = "existence"    # Check if property exists (OPTIONAL or FILTER EXISTS)


@dataclass
class SimplifiedFilter:
    """
    LLM-friendly filter specification.

    This is a higher-level representation that the LLM can easily generate.
    It uses natural property names rather than SPARQL predicates.

    Examples:
        # Property filter
        SimplifiedFilter(property="pos", value="noun")

        # Gender filter
        SimplifiedFilter(property="gender", value="masculine")

        # Pattern filter on written representation
        SimplifiedFilter(pattern="a$", pattern_type="ends_with", target="written_rep")

        # Numeric range filter
        SimplifiedFilter(property="polarity_value", min_value=0.5, max_value=1.0)

        # Negated filter
        SimplifiedFilter(property="pos", value="verb", negate=True)

    Attributes:
        property: Property name (e.g., "pos", "gender", "polarity")
        value: Property value (e.g., "noun", "masculine", "Positive")
        pattern: Text pattern for pattern-based filters
        pattern_type: Type of pattern matching (regex, starts_with, ends_with, contains)
        target: Target for pattern filters (written_rep, definition, etc.)
        min_value: Minimum value for range filters
        max_value: Maximum value for range filters
        negate: Whether to negate the filter (NOT)
        optional: Whether filter should be in OPTIONAL block
        retrieve: Whether to retrieve the value in SELECT
    """
    # Property-based filter
    property: Optional[str] = None
    value: Optional[str] = None

    # Pattern-based filter
    pattern: Optional[str] = None
    pattern_type: Optional[str] = None  # regex, starts_with, ends_with, contains
    target: Optional[str] = None        # written_rep, definition, etc.

    # Range filter
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Modifiers
    negate: bool = False
    optional: bool = False
    retrieve: bool = False  # If True, add variable to SELECT

    def get_filter_type(self) -> SimplifiedFilterType:
        """Determine the type of this filter"""
        if self.pattern is not None:
            return SimplifiedFilterType.PATTERN
        elif self.min_value is not None or self.max_value is not None:
            return SimplifiedFilterType.RANGE
        elif self.property is not None and self.value is None and not self.retrieve:
            return SimplifiedFilterType.EXISTENCE
        else:
            return SimplifiedFilterType.PROPERTY

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Basic validation of the filter structure.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        filter_type = self.get_filter_type()

        if filter_type == SimplifiedFilterType.PROPERTY:
            if not self.property:
                errors.append("Property filter requires 'property' field")
            if not self.value and not self.retrieve:
                errors.append("Property filter requires 'value' or 'retrieve=True'")

        elif filter_type == SimplifiedFilterType.PATTERN:
            if not self.pattern:
                errors.append("Pattern filter requires 'pattern' field")
            if self.pattern_type and self.pattern_type not in ["regex", "starts_with", "ends_with", "contains"]:
                errors.append(f"Invalid pattern_type: {self.pattern_type}")

        elif filter_type == SimplifiedFilterType.RANGE:
            if not self.property:
                errors.append("Range filter requires 'property' field")
            if self.min_value is None and self.max_value is None:
                errors.append("Range filter requires 'min_value' and/or 'max_value'")

        elif filter_type == SimplifiedFilterType.EXISTENCE:
            # Existence filter requires property with retrieve=True
            # A filter with just property (no value, no retrieve) is invalid
            if not self.property:
                errors.append("Existence filter requires 'property' field")
            if not self.retrieve:
                errors.append("Property filter requires 'value' or 'retrieve=True'")

        return len(errors) == 0, errors

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SimplifiedFilter':
        """Create SimplifiedFilter from dictionary (e.g., from LLM JSON output)"""
        return cls(
            property=d.get("property"),
            value=d.get("value"),
            pattern=d.get("pattern"),
            pattern_type=d.get("pattern_type"),
            target=d.get("target"),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            negate=d.get("negate", False),
            optional=d.get("optional", False),
            retrieve=d.get("retrieve", False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        d = {}
        if self.property:
            d["property"] = self.property
        if self.value:
            d["value"] = self.value
        if self.pattern:
            d["pattern"] = self.pattern
        if self.pattern_type:
            d["pattern_type"] = self.pattern_type
        if self.target:
            d["target"] = self.target
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        if self.negate:
            d["negate"] = self.negate
        if self.optional:
            d["optional"] = self.optional
        if self.retrieve:
            d["retrieve"] = self.retrieve
        return d


# ============================================================================
# Filter Translator
# ============================================================================

# Default variable mappings for common targets
DEFAULT_TARGET_VARIABLES = {
    "lemma": "?lemma",
    "written_rep": "?wr",
    "definition": "?definition",
    "sense": "?sense",
    "word": "?word",
    "polarity_value": "?polarityValue",
    "polarity": "?polarityLabel",
    "emotion": "?emotion",
    "emotion_label": "?emotionLabel",
    # Dialect-specific
    "sicilian_wr": "?sicilianWR",
    "parmigiano_wr": "?parmigianoWR",
    "italian_wr": "?italianWR",
}

# Property to target variable mappings (which variable to filter on)
PROPERTY_TARGET_MAPPINGS = {
    # LiITA properties target the lemma
    "pos": "?lemma",
    "gender": "?lemma",
    "inflection_type": "?lemma",
    # Sentix properties
    "polarity": "?sentixLemma",
    "polarity_value": "?polarityValue",
    # ELIta properties
    "emotion": "?elitaLemma",
}


class FilterTranslator:
    """
    Translates simplified filters to FilterSpecs.

    Uses the PropertyRegistry to resolve property names to SPARQL predicates
    and handle value normalization.

    Example:
        translator = FilterTranslator("liita")

        simplified = SimplifiedFilter(property="pos", value="noun")
        filter_spec = translator.translate(simplified)
        # Returns FilterSpec with property_path="lila:hasPOS", value="lila:noun"
    """

    def __init__(self, resource_type: str = "liita"):
        """
        Initialize translator for a specific resource type.

        Args:
            resource_type: The resource type (liita, complit, sentix, elita, etc.)
        """
        self.resource_type = resource_type.lower()
        self.registry = PropertyRegistry()
        self.builder = FilterBuilder()

    def translate(self, simplified: SimplifiedFilter) -> Optional[FilterSpec]:
        """
        Translate a simplified filter to a FilterSpec.

        Args:
            simplified: The simplified filter to translate

        Returns:
            FilterSpec or None if translation fails
        """
        filter_type = simplified.get_filter_type()

        if filter_type == SimplifiedFilterType.PROPERTY:
            return self._translate_property_filter(simplified)
        elif filter_type == SimplifiedFilterType.PATTERN:
            return self._translate_pattern_filter(simplified)
        elif filter_type == SimplifiedFilterType.RANGE:
            return self._translate_range_filter(simplified)
        elif filter_type == SimplifiedFilterType.EXISTENCE:
            return self._translate_existence_filter(simplified)
        else:
            return None

    def translate_all(
        self,
        simplified_filters: List[SimplifiedFilter]
    ) -> Tuple[List[FilterSpec], List[str]]:
        """
        Translate multiple simplified filters.

        Args:
            simplified_filters: List of simplified filters

        Returns:
            Tuple of (list of FilterSpecs, list of warnings)
        """
        specs = []
        warnings = []

        for i, sf in enumerate(simplified_filters):
            # Validate first
            is_valid, errors = sf.validate()
            if not is_valid:
                for error in errors:
                    warnings.append(f"Filter {i}: {error}")
                continue

            # Translate
            spec = self.translate(sf)
            if spec:
                specs.append(spec)
            else:
                warnings.append(f"Filter {i}: Could not translate filter")

        return specs, warnings

    def _translate_property_filter(self, sf: SimplifiedFilter) -> Optional[FilterSpec]:
        """Translate a property-based filter"""
        prop_name = sf.property.lower()

        # Special case: emotion filter needs EQUALS on ?emotion variable
        # (the ELItaLinkingPattern already generates the triple pattern)
        if prop_name == "emotion":
            emotion_value = sf.value
            if not emotion_value.startswith("elita:"):
                emotion_value = f"elita:{emotion_value}"
            return FilterSpec(
                target_variable="?emotion",
                filter_type=FilterType.EQUALS,
                value=emotion_value,
                negate=sf.negate,
                optional=sf.optional
            )

        # Get property info from registry
        prop_info = self.registry.get_property(self.resource_type, prop_name)

        if prop_info:
            # Use registry to get predicate and normalize value
            target_var = PROPERTY_TARGET_MAPPINGS.get(prop_name, "?lemma")

            if sf.retrieve:
                # Create retrieval filter (PROPERTY_EQUALS_VAR)
                return FilterSpec(
                    target_variable=target_var,
                    filter_type=FilterType.PROPERTY_EQUALS_VAR,
                    property_path=prop_info.predicate,
                    value_variable=f"?{prop_name}",
                    optional=sf.optional
                )
            else:
                # Create value filter (PROPERTY_EQUALS)
                try:
                    normalized_value = prop_info.normalize_value(sf.value)
                except ValueError:
                    # Invalid value - use as-is with prefix
                    normalized_value = f"{prop_info.value_prefix}{sf.value}" if prop_info.value_prefix else sf.value
                return FilterSpec(
                    target_variable=target_var,
                    filter_type=FilterType.PROPERTY_EQUALS,
                    property_path=prop_info.predicate,
                    value=normalized_value,
                    negate=sf.negate,
                    optional=sf.optional
                )
        else:
            # Property not in registry - try using FilterBuilder convenience methods
            return self._translate_with_builder(sf)

    def _translate_with_builder(self, sf: SimplifiedFilter) -> Optional[FilterSpec]:
        """Try to translate using FilterBuilder convenience methods"""
        prop_name = sf.property.lower() if sf.property else ""

        # Map property names to FilterBuilder methods
        if prop_name == "pos":
            return self.builder.pos_filter("?lemma", sf.value, retrieve=sf.retrieve)
        elif prop_name == "gender":
            return self.builder.gender_filter("?lemma", sf.value, retrieve=sf.retrieve)
        elif prop_name == "inflection_type":
            return self.builder.inflection_filter("?lemma", sf.value, retrieve=sf.retrieve)
        elif prop_name == "polarity":
            return self.builder.polarity_filter("?sentixLemma", sf.value, retrieve=sf.retrieve)
        elif prop_name == "emotion":
            # For emotion, we need an EQUALS filter on the ?emotion variable
            # (the triple pattern is already generated by ELItaLinkingPattern)
            emotion_value = sf.value
            if not emotion_value.startswith("elita:"):
                emotion_value = f"elita:{emotion_value}"
            return FilterSpec(
                target_variable="?emotion",
                filter_type=FilterType.EQUALS,
                value=emotion_value
            )
        else:
            return None

    def _translate_pattern_filter(self, sf: SimplifiedFilter) -> Optional[FilterSpec]:
        """Translate a pattern-based filter"""
        # Determine target variable
        target = sf.target or "written_rep"
        target_var = DEFAULT_TARGET_VARIABLES.get(target, f"?{target}")

        # Determine pattern type
        pattern_type = sf.pattern_type or "regex"

        # Clean up the pattern to avoid mixing regex anchors with string functions
        # e.g., "^infra" with starts_with should become just "infra"
        pattern = sf.pattern
        if pattern_type == "starts_with" and pattern.startswith("^"):
            pattern = pattern[1:]  # Remove leading ^
        elif pattern_type == "ends_with" and pattern.endswith("$"):
            pattern = pattern[:-1]  # Remove trailing $
        elif pattern_type == "contains":
            # Remove both anchors if present
            if pattern.startswith("^"):
                pattern = pattern[1:]
            if pattern.endswith("$"):
                pattern = pattern[:-1]

        # Use builder method
        spec = self.builder.written_rep_pattern(target_var, pattern, pattern_type)

        # Apply modifiers
        if sf.negate:
            spec.negate = True
        if sf.optional:
            spec.optional = True

        return spec

    def _translate_range_filter(self, sf: SimplifiedFilter) -> Optional[FilterSpec]:
        """Translate a range-based filter"""
        prop_name = sf.property.lower() if sf.property else ""

        # Determine target variable based on property
        target_var = DEFAULT_TARGET_VARIABLES.get(prop_name)
        if not target_var:
            # Try to construct from property name
            target_var = f"?{prop_name}"

        specs = []

        # Create min filter if specified
        if sf.min_value is not None:
            specs.append(FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.GREATER_EQUAL,
                value=str(sf.min_value),
                optional=sf.optional
            ))

        # Create max filter if specified
        if sf.max_value is not None:
            specs.append(FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.LESS_EQUAL,
                value=str(sf.max_value),
                optional=sf.optional
            ))

        # For range filters, we return the first spec
        # (the caller should handle multiple specs for full range support)
        return specs[0] if specs else None

    def _translate_range_filters(self, sf: SimplifiedFilter) -> List[FilterSpec]:
        """Translate a range filter to potentially multiple FilterSpecs"""
        prop_name = sf.property.lower() if sf.property else ""
        target_var = DEFAULT_TARGET_VARIABLES.get(prop_name, f"?{prop_name}")

        specs = []

        if sf.min_value is not None:
            specs.append(FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.GREATER_EQUAL,
                value=str(sf.min_value),
                optional=sf.optional
            ))

        if sf.max_value is not None:
            specs.append(FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.LESS_EQUAL,
                value=str(sf.max_value),
                optional=sf.optional
            ))

        return specs

    def _translate_existence_filter(self, sf: SimplifiedFilter) -> Optional[FilterSpec]:
        """Translate an existence filter (retrieve only)"""
        # This is essentially the same as property filter with retrieve=True
        sf_copy = SimplifiedFilter(
            property=sf.property,
            retrieve=True,
            optional=sf.optional
        )
        return self._translate_property_filter(sf_copy)


# ============================================================================
# Filter Validator
# ============================================================================

class FilterValidator:
    """
    Validates LLM-generated filters against the PropertyRegistry.

    Checks:
    - Property names are valid for the resource type
    - Values are in allowed_values (if defined)
    - Pattern types are valid
    - Filter structure is correct
    """

    def __init__(self):
        self.registry = PropertyRegistry()

    def validate(
        self,
        filters: List[Dict[str, Any]],
        resource_type: str
    ) -> Tuple[List[SimplifiedFilter], List[str]]:
        """
        Validate a list of filter dictionaries from LLM output.

        Args:
            filters: List of filter dictionaries
            resource_type: The resource type being queried

        Returns:
            Tuple of (valid_filters, warnings)
        """
        valid_filters = []
        warnings = []

        for i, f_dict in enumerate(filters):
            # Convert to SimplifiedFilter
            try:
                sf = SimplifiedFilter.from_dict(f_dict)
            except Exception as e:
                warnings.append(f"Filter {i}: Invalid structure - {str(e)}")
                continue

            # Basic structural validation
            is_valid, errors = sf.validate()
            if not is_valid:
                for error in errors:
                    warnings.append(f"Filter {i}: {error}")
                continue

            # Semantic validation
            semantic_warnings = self._validate_semantics(sf, resource_type, i)
            warnings.extend(semantic_warnings)

            # If no critical errors, add to valid filters
            # (semantic warnings don't prevent the filter from being used)
            valid_filters.append(sf)

        return valid_filters, warnings

    def _validate_semantics(
        self,
        sf: SimplifiedFilter,
        resource_type: str,
        index: int
    ) -> List[str]:
        """Validate semantic correctness of a filter"""
        warnings = []
        prefix = f"Filter {index}"

        if sf.property:
            # Check if property exists for resource type
            prop_info = self.registry.get_property(resource_type, sf.property)

            if not prop_info:
                # Property not found - might still work with builder
                # but warn the user
                warnings.append(
                    f"{prefix}: Property '{sf.property}' not in registry for "
                    f"'{resource_type}' - using default handling"
                )
            elif sf.value and prop_info.allowed_values:
                # Check if value is valid
                # Normalize value for comparison
                raw_value = sf.value.split(":")[-1] if ":" in sf.value else sf.value
                if raw_value.lower() not in [v.lower() for v in prop_info.allowed_values]:
                    warnings.append(
                        f"{prefix}: Value '{sf.value}' may not be valid for "
                        f"'{sf.property}'. Allowed: {prop_info.allowed_values}"
                    )

        if sf.pattern_type:
            valid_types = ["regex", "starts_with", "ends_with", "contains"]
            if sf.pattern_type not in valid_types:
                warnings.append(
                    f"{prefix}: Invalid pattern_type '{sf.pattern_type}'. "
                    f"Valid types: {valid_types}"
                )

        return warnings


# ============================================================================
# Convenience Functions
# ============================================================================

def translate_llm_filters(
    filter_dicts: List[Dict[str, Any]],
    resource_type: str = "liita"
) -> Tuple[List[FilterSpec], List[str]]:
    """
    Convenience function to validate and translate LLM filter output.

    Args:
        filter_dicts: List of filter dictionaries from LLM
        resource_type: The resource type being queried

    Returns:
        Tuple of (list of FilterSpecs, list of warnings)
    """
    # Validate
    validator = FilterValidator()
    valid_filters, warnings = validator.validate(filter_dicts, resource_type)

    # Translate
    translator = FilterTranslator(resource_type)
    specs = []

    for sf in valid_filters:
        spec = translator.translate(sf)
        if spec:
            specs.append(spec)

            # Handle range filters (may produce multiple specs)
            if sf.get_filter_type() == SimplifiedFilterType.RANGE:
                additional_specs = translator._translate_range_filters(sf)
                # Add any specs we missed (e.g., max when we only got min)
                for s in additional_specs:
                    if s not in specs:
                        specs.append(s)

    return specs, warnings


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SimplifiedFilter",
    "SimplifiedFilterType",
    "FilterTranslator",
    "FilterValidator",
    "translate_llm_filters",
    "DEFAULT_TARGET_VARIABLES",
    "PROPERTY_TARGET_MAPPINGS",
]


# ============================================================================
# Example / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FILTER LANGUAGE DEMONSTRATION")
    print("=" * 70)

    # Example 1: Property filter
    print("\nExample 1: Property Filter (POS = noun)")
    print("-" * 40)
    sf1 = SimplifiedFilter(property="pos", value="noun")
    translator = FilterTranslator("liita")
    spec1 = translator.translate(sf1)
    print(f"Input: {sf1.to_dict()}")
    print(f"Output: FilterSpec(")
    print(f"  target_variable={spec1.target_variable},")
    print(f"  filter_type={spec1.filter_type},")
    print(f"  property_path={spec1.property_path},")
    print(f"  value={spec1.value}")
    print(f")")

    # Example 2: Gender filter
    print("\nExample 2: Gender Filter")
    print("-" * 40)
    sf2 = SimplifiedFilter(property="gender", value="masculine")
    spec2 = translator.translate(sf2)
    print(f"Input: {sf2.to_dict()}")
    print(f"Output: property_path={spec2.property_path}, value={spec2.value}")

    # Example 3: Pattern filter
    print("\nExample 3: Pattern Filter (ends with 'a')")
    print("-" * 40)
    sf3 = SimplifiedFilter(pattern="a$", pattern_type="ends_with", target="written_rep")
    spec3 = translator.translate(sf3)
    print(f"Input: {sf3.to_dict()}")
    print(f"Output: filter_type={spec3.filter_type}, value={spec3.value}")

    # Example 4: Range filter
    print("\nExample 4: Range Filter (polarity_value >= 0.5)")
    print("-" * 40)
    sf4 = SimplifiedFilter(property="polarity_value", min_value=0.5)
    translator_sentix = FilterTranslator("sentix")
    spec4 = translator_sentix.translate(sf4)
    print(f"Input: {sf4.to_dict()}")
    print(f"Output: filter_type={spec4.filter_type}, value={spec4.value}")

    # Example 5: Full LLM output translation
    print("\nExample 5: Full LLM Output Translation")
    print("-" * 40)
    llm_filters = [
        {"property": "pos", "value": "noun"},
        {"property": "gender", "value": "masculine"},
        {"pattern": "a$", "pattern_type": "ends_with"}
    ]
    print(f"LLM Output: {llm_filters}")

    specs, warnings = translate_llm_filters(llm_filters, "liita")
    print(f"\nTranslated to {len(specs)} FilterSpecs:")
    for i, spec in enumerate(specs):
        print(f"  {i+1}. {spec.filter_type.value}: {spec.property_path or spec.value}")

    if warnings:
        print(f"\nWarnings: {warnings}")

    # Example 6: Validation with invalid input
    print("\nExample 6: Validation with Invalid Input")
    print("-" * 40)
    invalid_filters = [
        {"property": "pos", "value": "invalid_pos"},
        {"property": "unknown_property", "value": "test"},
        {"pattern": "test", "pattern_type": "invalid_type"}
    ]
    print(f"Input: {invalid_filters}")

    validator = FilterValidator()
    valid, warnings = validator.validate(invalid_filters, "liita")
    print(f"Valid filters: {len(valid)}")
    print(f"Warnings:")
    for w in warnings:
        print(f"  - {w}")

    print("\n" + "=" * 70)
    print("Filter language ready for LLM integration!")
    print("=" * 70)
