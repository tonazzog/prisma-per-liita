"""
PRISMA v2 Filter System
=======================

Flexible filter specification and rendering for SPARQL queries.

This module provides:
- FilterSpec: Declarative filter specifications
- FilterType: Supported filter operations
- FilterRenderer: Converts FilterSpecs to SPARQL clauses

The filter system separates filter logic from pattern structure,
enabling dynamic filter generation based on user intent.

Example:
    from prisma_v2.filter_system import FilterSpec, FilterType, FilterRenderer

    # Create filter specifications
    filters = [
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS,
            property_path="lila:hasPOS",
            value="lila:noun"
        ),
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS,
            property_path="lila:hasGender",
            value="lila:masculine"
        ),
        FilterSpec(
            target_variable="?wr",
            filter_type=FilterType.REGEX,
            value="a$"
        )
    ]

    # Render to SPARQL
    renderer = FilterRenderer()
    property_patterns, filter_clauses = renderer.render(filters)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
from enum import Enum


# ============================================================================
# Filter Types
# ============================================================================

class FilterType(Enum):
    """
    Supported filter operations.

    These map to different SPARQL constructs:
    - PROPERTY_EQUALS: Adds triple pattern with specific value
    - PROPERTY_EQUALS_VAR: Adds triple pattern with variable (for retrieval)
    - EQUALS: FILTER(?var = value)
    - EQUALS_LITERAL: FILTER(str(?var) = "value")
    - REGEX: FILTER(regex(str(?var), "pattern"))
    - STARTS_WITH: FILTER(strstarts(str(?var), "value"))
    - ENDS_WITH: FILTER(strends(str(?var), "value"))
    - CONTAINS: FILTER(contains(str(?var), "value"))
    - GREATER_THAN: FILTER(?var > value)
    - LESS_THAN: FILTER(?var < value)
    - GREATER_EQUAL: FILTER(?var >= value)
    - LESS_EQUAL: FILTER(?var <= value)
    - IN: FILTER(?var IN (value1, value2, ...))
    - NOT_EQUALS: FILTER(?var != value)
    - LANG: FILTER(lang(?var) = "value")
    """
    # Property-based filters (add triple patterns)
    PROPERTY_EQUALS = "property_equals"      # ?s prop value .
    PROPERTY_EQUALS_VAR = "property_equals_var"  # ?s prop ?var . (for retrieval)

    # Value comparison filters
    EQUALS = "equals"                        # FILTER(?var = value)
    EQUALS_LITERAL = "equals_literal"        # FILTER(str(?var) = "value")
    NOT_EQUALS = "not_equals"                # FILTER(?var != value)

    # String filters
    REGEX = "regex"                          # FILTER(regex(str(?var), "pattern"))
    STARTS_WITH = "starts_with"              # FILTER(strstarts(str(?var), "value"))
    ENDS_WITH = "ends_with"                  # FILTER(strends(str(?var), "value"))
    CONTAINS = "contains"                    # FILTER(contains(str(?var), "value"))

    # Numeric comparison filters
    GREATER_THAN = "greater_than"            # FILTER(?var > value)
    LESS_THAN = "less_than"                  # FILTER(?var < value)
    GREATER_EQUAL = "greater_equal"          # FILTER(?var >= value)
    LESS_EQUAL = "less_equal"                # FILTER(?var <= value)

    # Set filters
    IN = "in"                                # FILTER(?var IN (v1, v2, ...))

    # Language filter
    LANG = "lang"                            # FILTER(lang(?var) = "value")


# ============================================================================
# Filter Specification
# ============================================================================

@dataclass
class FilterSpec:
    """
    A declarative filter specification.

    Filters can be:
    1. Property-based: Add a triple pattern to access a property
    2. Value-based: Add a FILTER clause to constrain a variable

    Attributes:
        target_variable: The SPARQL variable to filter (e.g., "?lemma", "?wr")
        filter_type: The type of filter operation
        value: The filter value (literal, URI, or pattern)
        property_path: For property-based filters, the predicate path
        value_variable: For PROPERTY_EQUALS_VAR, the variable name for the value
        case_insensitive: For string filters, whether to ignore case
        negate: Whether to negate the filter (NOT)
        optional: Whether this filter should be in an OPTIONAL block

    Examples:
        # Filter by POS (property-based)
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS,
            property_path="lila:hasPOS",
            value="lila:noun"
        )

        # Filter written rep by regex
        FilterSpec(
            target_variable="?wr",
            filter_type=FilterType.REGEX,
            value="^casa"
        )

        # Retrieve gender (property access for display)
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS_VAR,
            property_path="lila:hasGender",
            value_variable="?gender"
        )
    """
    target_variable: str
    filter_type: FilterType
    value: Optional[str] = None
    property_path: Optional[str] = None
    value_variable: Optional[str] = None
    case_insensitive: bool = False
    negate: bool = False
    optional: bool = False

    def __post_init__(self):
        """Validate the filter specification"""
        # Ensure variable starts with ?
        if not self.target_variable.startswith("?"):
            self.target_variable = "?" + self.target_variable

        # Validate property-based filters have property_path
        if self.filter_type in [FilterType.PROPERTY_EQUALS, FilterType.PROPERTY_EQUALS_VAR]:
            if not self.property_path:
                raise ValueError(f"Property-based filter requires property_path")

        # Validate PROPERTY_EQUALS_VAR has value_variable
        if self.filter_type == FilterType.PROPERTY_EQUALS_VAR:
            if not self.value_variable:
                raise ValueError("PROPERTY_EQUALS_VAR requires value_variable")
            if not self.value_variable.startswith("?"):
                self.value_variable = "?" + self.value_variable

        # Validate value-based filters have value
        if self.filter_type not in [FilterType.PROPERTY_EQUALS_VAR]:
            if self.value is None:
                raise ValueError(f"Filter type {self.filter_type} requires a value")

    def get_output_variable(self) -> Optional[str]:
        """Return the output variable if this filter creates one"""
        if self.filter_type == FilterType.PROPERTY_EQUALS_VAR:
            return self.value_variable
        return None

    def requires_property_access(self) -> bool:
        """Check if this filter needs a property triple pattern"""
        return self.filter_type in [
            FilterType.PROPERTY_EQUALS,
            FilterType.PROPERTY_EQUALS_VAR
        ]


# ============================================================================
# Filter Renderer
# ============================================================================

class FilterRenderer:
    """
    Renders FilterSpecs to SPARQL clauses.

    Produces two outputs:
    1. Property accessor patterns (triple patterns)
    2. FILTER clauses

    Example:
        renderer = FilterRenderer()
        props, filters = renderer.render([
            FilterSpec("?lemma", FilterType.PROPERTY_EQUALS,
                      property_path="lila:hasPOS", value="lila:noun"),
            FilterSpec("?wr", FilterType.REGEX, value="a$")
        ])

        # props = "?lemma lila:hasPOS lila:noun ."
        # filters = 'FILTER(regex(str(?wr), "a$")) .'
    """

    def render(
        self,
        filter_specs: List[FilterSpec]
    ) -> Tuple[str, str, Set[str]]:
        """
        Render filter specifications to SPARQL.

        Args:
            filter_specs: List of filter specifications

        Returns:
            Tuple of (property_patterns, filter_clauses, output_variables)
            - property_patterns: Triple patterns for property access
            - filter_clauses: FILTER clauses
            - output_variables: Set of new variables created by filters
        """
        property_patterns = []
        filter_clauses = []
        output_variables = set()
        optional_patterns = []
        optional_filters = []

        for spec in filter_specs:
            if spec.optional:
                pattern, clause = self._render_single(spec)
                if pattern:
                    optional_patterns.append(pattern)
                if clause:
                    optional_filters.append(clause)
            else:
                pattern, clause = self._render_single(spec)
                if pattern:
                    property_patterns.append(pattern)
                if clause:
                    filter_clauses.append(clause)

            # Track output variables
            out_var = spec.get_output_variable()
            if out_var:
                output_variables.add(out_var)

        # Build OPTIONAL block if needed
        optional_block = ""
        if optional_patterns or optional_filters:
            optional_content = "\n    ".join(optional_patterns + optional_filters)
            optional_block = f"\n  OPTIONAL {{\n    {optional_content}\n  }}"

        # Combine results
        props_str = "\n  ".join(property_patterns)
        if optional_block:
            props_str += optional_block

        filters_str = "\n  ".join(filter_clauses)

        return props_str, filters_str, output_variables

    def _render_single(self, spec: FilterSpec) -> Tuple[Optional[str], Optional[str]]:
        """
        Render a single filter specification.

        Returns:
            Tuple of (property_pattern, filter_clause)
        """
        # Property-based filters
        if spec.filter_type == FilterType.PROPERTY_EQUALS:
            pattern = f"{spec.target_variable} {spec.property_path} {spec.value} ."
            return pattern, None

        elif spec.filter_type == FilterType.PROPERTY_EQUALS_VAR:
            pattern = f"{spec.target_variable} {spec.property_path} {spec.value_variable} ."
            return pattern, None

        # Value comparison filters
        elif spec.filter_type == FilterType.EQUALS:
            clause = f"FILTER({spec.target_variable} = {spec.value}) ."
            if spec.negate:
                clause = f"FILTER({spec.target_variable} != {spec.value}) ."
            return None, clause

        elif spec.filter_type == FilterType.EQUALS_LITERAL:
            clause = f'FILTER(str({spec.target_variable}) = "{spec.value}") .'
            if spec.negate:
                clause = f'FILTER(str({spec.target_variable}) != "{spec.value}") .'
            return None, clause

        elif spec.filter_type == FilterType.NOT_EQUALS:
            clause = f"FILTER({spec.target_variable} != {spec.value}) ."
            return None, clause

        # String filters
        elif spec.filter_type == FilterType.REGEX:
            flags = ', "i"' if spec.case_insensitive else ''
            clause = f'FILTER(regex(str({spec.target_variable}), "{spec.value}"{flags})) .'
            if spec.negate:
                clause = f'FILTER(!regex(str({spec.target_variable}), "{spec.value}"{flags})) .'
            return None, clause

        elif spec.filter_type == FilterType.STARTS_WITH:
            func = "strstarts"
            if spec.case_insensitive:
                clause = f'FILTER({func}(lcase(str({spec.target_variable})), lcase("{spec.value}"))) .'
            else:
                clause = f'FILTER({func}(str({spec.target_variable}), "{spec.value}")) .'
            if spec.negate:
                clause = clause.replace("FILTER(", "FILTER(!")
            return None, clause

        elif spec.filter_type == FilterType.ENDS_WITH:
            func = "strends"
            if spec.case_insensitive:
                clause = f'FILTER({func}(lcase(str({spec.target_variable})), lcase("{spec.value}"))) .'
            else:
                clause = f'FILTER({func}(str({spec.target_variable}), "{spec.value}")) .'
            if spec.negate:
                clause = clause.replace("FILTER(", "FILTER(!")
            return None, clause

        elif spec.filter_type == FilterType.CONTAINS:
            if spec.case_insensitive:
                clause = f'FILTER(contains(lcase(str({spec.target_variable})), lcase("{spec.value}"))) .'
            else:
                clause = f'FILTER(contains(str({spec.target_variable}), "{spec.value}")) .'
            if spec.negate:
                clause = clause.replace("FILTER(", "FILTER(!")
            return None, clause

        # Numeric comparison filters
        elif spec.filter_type == FilterType.GREATER_THAN:
            clause = f"FILTER({spec.target_variable} > {spec.value}) ."
            return None, clause

        elif spec.filter_type == FilterType.LESS_THAN:
            clause = f"FILTER({spec.target_variable} < {spec.value}) ."
            return None, clause

        elif spec.filter_type == FilterType.GREATER_EQUAL:
            clause = f"FILTER({spec.target_variable} >= {spec.value}) ."
            return None, clause

        elif spec.filter_type == FilterType.LESS_EQUAL:
            clause = f"FILTER({spec.target_variable} <= {spec.value}) ."
            return None, clause

        # Set filter
        elif spec.filter_type == FilterType.IN:
            clause = f"FILTER({spec.target_variable} IN ({spec.value})) ."
            if spec.negate:
                clause = f"FILTER({spec.target_variable} NOT IN ({spec.value})) ."
            return None, clause

        # Language filter
        elif spec.filter_type == FilterType.LANG:
            clause = f'FILTER(lang({spec.target_variable}) = "{spec.value}") .'
            return None, clause

        else:
            raise ValueError(f"Unknown filter type: {spec.filter_type}")


# ============================================================================
# Filter Builder (Convenience Methods)
# ============================================================================

class FilterBuilder:
    """
    Convenience builder for creating common filter specifications.

    Example:
        builder = FilterBuilder()
        filters = [
            builder.pos_filter("?lemma", "noun"),
            builder.gender_filter("?lemma", "masculine"),
            builder.written_rep_pattern("?wr", "^casa"),
        ]
    """

    # Common property paths
    POS_PATH = "lila:hasPOS"
    GENDER_PATH = "lila:hasGender"
    INFLECTION_PATH = "lila:hasInflectionType"
    WRITTEN_REP_PATH = "ontolex:writtenRep"

    # Common value prefixes
    LILA_PREFIX = "lila:"

    def pos_filter(
        self,
        target_var: str,
        pos: str,
        retrieve: bool = False
    ) -> FilterSpec:
        """
        Create a part-of-speech filter.

        Args:
            target_var: The lemma variable (e.g., "?lemma")
            pos: The POS value (e.g., "noun", "verb")
            retrieve: If True, also retrieve the POS value
        """
        # Normalize POS value
        if not pos.startswith(self.LILA_PREFIX):
            pos = f"{self.LILA_PREFIX}{pos}"

        if retrieve:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS_VAR,
                property_path=self.POS_PATH,
                value_variable="?pos"
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS,
                property_path=self.POS_PATH,
                value=pos
            )

    def gender_filter(
        self,
        target_var: str,
        gender: str,
        retrieve: bool = False
    ) -> FilterSpec:
        """
        Create a grammatical gender filter.

        Args:
            target_var: The lemma variable
            gender: The gender value (e.g., "masculine", "feminine")
            retrieve: If True, also retrieve the gender value
        """
        if not gender.startswith(self.LILA_PREFIX):
            gender = f"{self.LILA_PREFIX}{gender}"

        if retrieve:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS_VAR,
                property_path=self.GENDER_PATH,
                value_variable="?gender"
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS,
                property_path=self.GENDER_PATH,
                value=gender
            )

    def inflection_filter(
        self,
        target_var: str,
        inflection_type: str,
        retrieve: bool = False
    ) -> FilterSpec:
        """
        Create an inflection type filter.
        """
        if not inflection_type.startswith(self.LILA_PREFIX):
            inflection_type = f"{self.LILA_PREFIX}{inflection_type}"

        if retrieve:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS_VAR,
                property_path=self.INFLECTION_PATH,
                value_variable="?inflectionType"
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS,
                property_path=self.INFLECTION_PATH,
                value=inflection_type
            )

    def written_rep_pattern(
        self,
        target_var: str,
        pattern: str,
        pattern_type: str = "regex"
    ) -> FilterSpec:
        """
        Create a written representation pattern filter.

        Args:
            target_var: The written rep variable (e.g., "?wr")
            pattern: The pattern to match
            pattern_type: One of "regex", "starts_with", "ends_with", "contains", "equals"
        """
        type_map = {
            "regex": FilterType.REGEX,
            "starts_with": FilterType.STARTS_WITH,
            "ends_with": FilterType.ENDS_WITH,
            "contains": FilterType.CONTAINS,
            "equals": FilterType.EQUALS_LITERAL,
        }

        filter_type = type_map.get(pattern_type, FilterType.REGEX)

        # Clean up pattern to avoid mixing regex anchors with string functions
        # e.g., "^ante" with starts_with should become just "ante"
        clean_pattern = pattern
        if pattern_type == "starts_with" and pattern.startswith("^"):
            clean_pattern = pattern[1:]  # Remove leading ^
        elif pattern_type == "ends_with" and pattern.endswith("$"):
            clean_pattern = pattern[:-1]  # Remove trailing $
        elif pattern_type == "contains":
            # Remove both anchors if present
            if clean_pattern.startswith("^"):
                clean_pattern = clean_pattern[1:]
            if clean_pattern.endswith("$"):
                clean_pattern = clean_pattern[:-1]

        return FilterSpec(
            target_variable=target_var,
            filter_type=filter_type,
            value=clean_pattern
        )

    def exact_match(
        self,
        target_var: str,
        value: str,
        is_literal: bool = True
    ) -> FilterSpec:
        """
        Create an exact match filter.

        Args:
            target_var: The variable to filter
            value: The exact value to match
            is_literal: If True, use string comparison
        """
        if is_literal:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.EQUALS_LITERAL,
                value=value
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.EQUALS,
                value=value
            )

    def from_dict(self, spec_dict: Dict[str, Any]) -> FilterSpec:
        """
        Create a FilterSpec from a dictionary.

        Useful for deserializing from JSON (e.g., from intent analyzer).
        """
        filter_type = spec_dict.get("filter_type", spec_dict.get("type"))
        if isinstance(filter_type, str):
            filter_type = FilterType(filter_type)

        return FilterSpec(
            target_variable=spec_dict["target_variable"],
            filter_type=filter_type,
            value=spec_dict.get("value"),
            property_path=spec_dict.get("property_path"),
            value_variable=spec_dict.get("value_variable"),
            case_insensitive=spec_dict.get("case_insensitive", False),
            negate=spec_dict.get("negate", False),
            optional=spec_dict.get("optional", False)
        )

    # =========================================================================
    # Sentix (Affective) Filters
    # =========================================================================

    # Sentix property paths
    POLARITY_PATH = "marl:hasPolarity"
    POLARITY_VALUE_PATH = "marl:polarityValue"
    MARL_PREFIX = "marl:"

    def polarity_filter(
        self,
        target_var: str,
        polarity: str,
        retrieve: bool = False
    ) -> FilterSpec:
        """
        Create a sentiment polarity filter for Sentix.

        Args:
            target_var: The sentiment variable (e.g., "?sentiment")
            polarity: The polarity value ("Positive", "Negative", "Neutral")
            retrieve: If True, retrieve the polarity value
        """
        if not polarity.startswith(self.MARL_PREFIX):
            polarity = f"{self.MARL_PREFIX}{polarity}"

        if retrieve:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS_VAR,
                property_path=self.POLARITY_PATH,
                value_variable="?polarity"
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS,
                property_path=self.POLARITY_PATH,
                value=polarity
            )

    def polarity_value_min(
        self,
        target_var: str,
        min_value: float
    ) -> FilterSpec:
        """
        Create a minimum polarity value filter.

        Args:
            target_var: The polarity value variable (e.g., "?polarityValue")
            min_value: Minimum polarity value (-1 to 1)
        """
        return FilterSpec(
            target_variable=target_var,
            filter_type=FilterType.GREATER_EQUAL,
            value=str(min_value)
        )

    def polarity_value_max(
        self,
        target_var: str,
        max_value: float
    ) -> FilterSpec:
        """
        Create a maximum polarity value filter.

        Args:
            target_var: The polarity value variable (e.g., "?polarityValue")
            max_value: Maximum polarity value (-1 to 1)
        """
        return FilterSpec(
            target_variable=target_var,
            filter_type=FilterType.LESS_EQUAL,
            value=str(max_value)
        )

    # =========================================================================
    # ELIta (Emotion) Filters
    # =========================================================================

    EMOTION_PATH = "elita:hasEmotion"
    ELITA_PREFIX = "elita:"

    def emotion_filter(
        self,
        target_var: str,
        emotion: str,
        retrieve: bool = False
    ) -> FilterSpec:
        """
        Create an emotion filter for ELIta.

        Args:
            target_var: The emotion variable (e.g., "?emotionAssoc")
            emotion: The emotion value (e.g., "Gioia", "Tristezza")
            retrieve: If True, retrieve the emotion value
        """
        if not emotion.startswith(self.ELITA_PREFIX):
            emotion = f"{self.ELITA_PREFIX}{emotion}"

        if retrieve:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS_VAR,
                property_path=self.EMOTION_PATH,
                value_variable="?emotion"
            )
        else:
            return FilterSpec(
                target_variable=target_var,
                filter_type=FilterType.PROPERTY_EQUALS,
                property_path=self.EMOTION_PATH,
                value=emotion
            )

    def emotion_in_list(
        self,
        target_var: str,
        emotions: List[str]
    ) -> FilterSpec:
        """
        Create a filter for multiple emotions (IN clause).

        Args:
            target_var: The emotion variable
            emotions: List of emotion values
        """
        # Format as comma-separated prefixed values
        prefixed = [f"{self.ELITA_PREFIX}{e}" if not e.startswith(self.ELITA_PREFIX) else e
                    for e in emotions]
        values_str = ", ".join(prefixed)

        return FilterSpec(
            target_variable=target_var,
            filter_type=FilterType.IN,
            value=values_str
        )

    # =========================================================================
    # Definition/Text Pattern Filters (CompL-it)
    # =========================================================================

    def definition_pattern(
        self,
        target_var: str,
        pattern: str,
        pattern_type: str = "contains"
    ) -> FilterSpec:
        """
        Create a definition text pattern filter for CompL-it.

        Args:
            target_var: The definition variable (e.g., "?definition")
            pattern: The pattern to match
            pattern_type: One of "regex", "starts_with", "ends_with", "contains"
        """
        type_map = {
            "regex": FilterType.REGEX,
            "starts_with": FilterType.STARTS_WITH,
            "ends_with": FilterType.ENDS_WITH,
            "contains": FilterType.CONTAINS,
        }

        filter_type = type_map.get(pattern_type, FilterType.CONTAINS)

        return FilterSpec(
            target_variable=target_var,
            filter_type=filter_type,
            value=pattern
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "FilterType",
    "FilterSpec",
    "FilterRenderer",
    "FilterBuilder",
]


# ============================================================================
# Example / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FILTER SYSTEM DEMONSTRATION")
    print("=" * 70)

    builder = FilterBuilder()
    renderer = FilterRenderer()

    # Example 1: Simple POS filter
    print("\nExample 1: Filter by POS (noun)")
    print("-" * 40)
    filters = [
        builder.pos_filter("?lemma", "noun")
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Property patterns:\n  {props}")
    print(f"Filter clauses:\n  {clauses}")

    # Example 2: POS + Gender + Written Rep pattern
    print("\n\nExample 2: Masculine nouns ending in 'a'")
    print("-" * 40)
    filters = [
        builder.pos_filter("?lemma", "noun"),
        builder.gender_filter("?lemma", "masculine"),
        builder.written_rep_pattern("?wr", "a$", "regex")
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Property patterns:\n  {props}")
    print(f"Filter clauses:\n  {clauses}")

    # Example 3: Property access for retrieval
    print("\n\nExample 3: Retrieve gender of nouns")
    print("-" * 40)
    filters = [
        builder.pos_filter("?lemma", "noun"),
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS_VAR,
            property_path="lila:hasGender",
            value_variable="?gender"
        )
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Property patterns:\n  {props}")
    print(f"Output variables: {out_vars}")

    # Example 4: Complex filter with OPTIONAL
    print("\n\nExample 4: Nouns with optional gender retrieval")
    print("-" * 40)
    filters = [
        builder.pos_filter("?lemma", "noun"),
        FilterSpec(
            target_variable="?lemma",
            filter_type=FilterType.PROPERTY_EQUALS_VAR,
            property_path="lila:hasGender",
            value_variable="?gender",
            optional=True
        )
    ]
    props, clauses, out_vars = renderer.render(filters)
    print(f"Property patterns:\n{props}")

    print("\n" + "=" * 70)
    print("Filter system ready for integration!")
    print("=" * 70)
