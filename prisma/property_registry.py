"""
PRISMA v2 Property Registry
============================

Configuration of filterable properties for each resource type.

This module defines:
- What properties can be filtered on each resource type
- How property values should be formatted
- Mappings from user-friendly names to SPARQL predicates

The registry enables dynamic filter generation without hardcoding
property knowledge in pattern tools.

Example:
    from prisma_v2.property_registry import PropertyRegistry

    registry = PropertyRegistry()

    # Get available properties for LiITA lemmas
    props = registry.get_properties("liita_lemma")
    # Returns: {"pos": PropertyInfo(...), "gender": PropertyInfo(...), ...}

    # Validate and normalize a property value
    normalized = registry.normalize_value("liita_lemma", "pos", "noun")
    # Returns: "lila:noun"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


# ============================================================================
# Property Value Types
# ============================================================================

class ValueType(Enum):
    """Type of property value"""
    URI = "uri"           # Full URI or prefixed URI (e.g., lila:noun)
    LITERAL = "literal"   # String literal
    NUMERIC = "numeric"   # Number
    BOOLEAN = "boolean"   # True/False
    LANG_LITERAL = "lang_literal"  # Language-tagged literal


# ============================================================================
# Property Information
# ============================================================================

@dataclass
class PropertyInfo:
    """
    Information about a filterable property.

    Attributes:
        name: User-friendly property name (e.g., "pos", "gender")
        predicate: SPARQL predicate path (e.g., "lila:hasPOS")
        value_type: Type of values this property accepts
        value_prefix: Prefix to add to values (e.g., "lila:")
        allowed_values: Optional list of allowed values (for enums)
        description: Human-readable description
        aliases: Alternative names for this property
        required_prefix: Required prefix in SPARQL (e.g., "lila")
    """
    name: str
    predicate: str
    value_type: ValueType
    value_prefix: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    required_prefix: Optional[str] = None

    def normalize_value(self, value: str) -> str:
        """
        Normalize a property value for SPARQL.

        - Adds prefix if needed
        - Validates against allowed values if defined
        """
        # Handle already-prefixed values
        if self.value_prefix and value.startswith(self.value_prefix):
            normalized = value
        elif self.value_prefix and ":" in value:
            # Already has a different prefix
            normalized = value
        elif self.value_prefix:
            normalized = f"{self.value_prefix}{value}"
        else:
            normalized = value

        # Validate against allowed values if defined
        if self.allowed_values:
            # Check both normalized and raw value
            raw_value = value.split(":")[-1] if ":" in value else value
            if raw_value not in self.allowed_values:
                valid = ", ".join(self.allowed_values)
                raise ValueError(
                    f"Invalid value '{value}' for property '{self.name}'. "
                    f"Allowed values: {valid}"
                )

        return normalized

    def get_variable_name(self) -> str:
        """Generate a SPARQL variable name for this property"""
        return f"?{self.name}"


# ============================================================================
# Resource Property Definitions
# ============================================================================

# LiITA Lemma Properties
LIITA_LEMMA_PROPERTIES = {
    "pos": PropertyInfo(
        name="pos",
        predicate="lila:hasPOS",
        value_type=ValueType.URI,
        value_prefix="lila:",
        allowed_values=[
            "noun", "verb", "adjective", "adverb", "pronoun",
            "preposition", "conjunction", "interjection", "article",
            "numeral", "participle"
        ],
        description="Part of speech",
        aliases=["part_of_speech", "partofspeech", "partOfSpeech"],
        required_prefix="lila"
    ),
    "gender": PropertyInfo(
        name="gender",
        predicate="lila:hasGender",
        value_type=ValueType.URI,
        value_prefix="lila:",
        allowed_values=["masculine", "feminine", "neuter", "common"],
        description="Grammatical gender",
        aliases=["grammatical_gender"],
        required_prefix="lila"
    ),
    "inflection_type": PropertyInfo(
        name="inflection_type",
        predicate="lila:hasInflectionType",
        value_type=ValueType.URI,
        value_prefix="lila:",
        description="Inflection/declension type",
        aliases=["inflection", "declension"],
        required_prefix="lila"
    ),
    "written_rep": PropertyInfo(
        name="written_rep",
        predicate="ontolex:writtenRep",
        value_type=ValueType.LITERAL,
        description="Written representation (lemma form)",
        aliases=["writtenRep", "lemma_form", "form"],
        required_prefix="ontolex"
    ),
}

# CompL-it Word Properties
COMPLIT_WORD_PROPERTIES = {
    "pos": PropertyInfo(
        name="pos",
        predicate="lexinfo:partOfSpeech",
        value_type=ValueType.URI,
        description="Part of speech (via blank node with rdfs:label)",
        aliases=["part_of_speech"],
        required_prefix="lexinfo"
    ),
    "written_rep": PropertyInfo(
        name="written_rep",
        predicate="ontolex:writtenRep",
        value_type=ValueType.LITERAL,
        description="Written representation",
        aliases=["lemma"],
        required_prefix="ontolex"
    ),
}

# Sicilian Lemma Properties
SICILIAN_LEMMA_PROPERTIES = {
    "pos": PropertyInfo(
        name="pos",
        predicate="lila:hasPOS",
        value_type=ValueType.URI,
        value_prefix="lila:",
        allowed_values=["noun", "verb", "adjective", "adverb"],
        description="Part of speech",
        required_prefix="lila"
    ),
    "written_rep": PropertyInfo(
        name="written_rep",
        predicate="ontolex:writtenRep",
        value_type=ValueType.LITERAL,
        description="Written representation in Sicilian",
        required_prefix="ontolex"
    ),
}

# Parmigiano Lemma Properties
PARMIGIANO_LEMMA_PROPERTIES = {
    "pos": PropertyInfo(
        name="pos",
        predicate="lila:hasPOS",
        value_type=ValueType.URI,
        value_prefix="lila:",
        allowed_values=["noun", "verb", "adjective", "adverb"],
        description="Part of speech",
        required_prefix="lila"
    ),
    "written_rep": PropertyInfo(
        name="written_rep",
        predicate="ontolex:writtenRep",
        value_type=ValueType.LITERAL,
        description="Written representation in Parmigiano",
        required_prefix="ontolex"
    ),
}

# Sentix Properties
SENTIX_PROPERTIES = {
    "polarity": PropertyInfo(
        name="polarity",
        predicate="marl:hasPolarity",
        value_type=ValueType.URI,
        value_prefix="marl:",
        allowed_values=["Positive", "Negative", "Neutral"],
        description="Sentiment polarity",
        required_prefix="marl"
    ),
    "polarity_value": PropertyInfo(
        name="polarity_value",
        predicate="marl:polarityValue",
        value_type=ValueType.NUMERIC,
        description="Numeric polarity value (-1 to 1)",
    ),
}

# ELIta Properties
ELITA_PROPERTIES = {
    "emotion": PropertyInfo(
        name="emotion",
        predicate="elita:hasEmotion",
        value_type=ValueType.URI,
        description="Associated emotion",
        required_prefix="elita"
    ),
}


# ============================================================================
# Property Registry
# ============================================================================

class PropertyRegistry:
    """
    Central registry of filterable properties for all resource types.

    Provides:
    - Property lookup by resource type
    - Value normalization and validation
    - Alias resolution
    """

    def __init__(self):
        """Initialize the registry with default property definitions"""
        self._registries: Dict[str, Dict[str, PropertyInfo]] = {
            "liita_lemma": LIITA_LEMMA_PROPERTIES,
            "liita": LIITA_LEMMA_PROPERTIES,  # Alias
            "complit_word": COMPLIT_WORD_PROPERTIES,
            "complit": COMPLIT_WORD_PROPERTIES,  # Alias
            "sicilian_lemma": SICILIAN_LEMMA_PROPERTIES,
            "sicilian": SICILIAN_LEMMA_PROPERTIES,  # Alias
            "parmigiano_lemma": PARMIGIANO_LEMMA_PROPERTIES,
            "parmigiano": PARMIGIANO_LEMMA_PROPERTIES,  # Alias
            "sentix": SENTIX_PROPERTIES,
            "elita": ELITA_PROPERTIES,
        }

        # Build alias index
        self._alias_index: Dict[str, Dict[str, str]] = {}
        for resource_type, props in self._registries.items():
            self._alias_index[resource_type] = {}
            for prop_name, prop_info in props.items():
                # Map primary name
                self._alias_index[resource_type][prop_name] = prop_name
                # Map aliases
                for alias in prop_info.aliases:
                    self._alias_index[resource_type][alias.lower()] = prop_name

    def get_properties(self, resource_type: str) -> Dict[str, PropertyInfo]:
        """
        Get all filterable properties for a resource type.

        Args:
            resource_type: The resource type (e.g., "liita_lemma", "complit")

        Returns:
            Dictionary of property name to PropertyInfo
        """
        resource_type = resource_type.lower()
        if resource_type not in self._registries:
            raise ValueError(f"Unknown resource type: {resource_type}")
        return self._registries[resource_type]

    def get_property(
        self,
        resource_type: str,
        property_name: str
    ) -> Optional[PropertyInfo]:
        """
        Get a specific property by name or alias.

        Args:
            resource_type: The resource type
            property_name: The property name or alias

        Returns:
            PropertyInfo if found, None otherwise
        """
        resource_type = resource_type.lower()
        property_name = property_name.lower()

        if resource_type not in self._alias_index:
            return None

        # Resolve alias
        canonical_name = self._alias_index[resource_type].get(property_name)
        if not canonical_name:
            return None

        return self._registries[resource_type].get(canonical_name)

    def normalize_value(
        self,
        resource_type: str,
        property_name: str,
        value: str
    ) -> str:
        """
        Normalize a property value for SPARQL.

        Args:
            resource_type: The resource type
            property_name: The property name
            value: The value to normalize

        Returns:
            Normalized value string
        """
        prop = self.get_property(resource_type, property_name)
        if not prop:
            raise ValueError(
                f"Unknown property '{property_name}' for resource '{resource_type}'"
            )
        return prop.normalize_value(value)

    def get_predicate(
        self,
        resource_type: str,
        property_name: str
    ) -> Optional[str]:
        """
        Get the SPARQL predicate for a property.

        Args:
            resource_type: The resource type
            property_name: The property name

        Returns:
            Predicate string (e.g., "lila:hasPOS")
        """
        prop = self.get_property(resource_type, property_name)
        return prop.predicate if prop else None

    def get_required_prefixes(
        self,
        resource_type: str,
        properties: List[str]
    ) -> Set[str]:
        """
        Get required SPARQL prefixes for a set of properties.

        Args:
            resource_type: The resource type
            properties: List of property names

        Returns:
            Set of required prefix names
        """
        prefixes = set()
        for prop_name in properties:
            prop = self.get_property(resource_type, prop_name)
            if prop and prop.required_prefix:
                prefixes.add(prop.required_prefix)
        return prefixes

    def list_resource_types(self) -> List[str]:
        """List all supported resource types"""
        # Return unique types (excluding aliases)
        return ["liita_lemma", "complit_word", "sicilian_lemma",
                "parmigiano_lemma", "sentix", "elita"]

    def list_properties(self, resource_type: str) -> List[str]:
        """List all property names for a resource type"""
        props = self.get_properties(resource_type)
        return list(props.keys())


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ValueType",
    "PropertyInfo",
    "PropertyRegistry",
    "LIITA_LEMMA_PROPERTIES",
    "COMPLIT_WORD_PROPERTIES",
    "SICILIAN_LEMMA_PROPERTIES",
    "PARMIGIANO_LEMMA_PROPERTIES",
    "SENTIX_PROPERTIES",
    "ELITA_PROPERTIES",
]


# ============================================================================
# Example / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROPERTY REGISTRY DEMONSTRATION")
    print("=" * 70)

    registry = PropertyRegistry()

    # List resource types
    print("\nSupported resource types:")
    for rt in registry.list_resource_types():
        print(f"  - {rt}")

    # Show LiITA properties
    print("\n\nLiITA Lemma Properties:")
    print("-" * 40)
    for name, prop in registry.get_properties("liita_lemma").items():
        print(f"  {name}:")
        print(f"    Predicate: {prop.predicate}")
        print(f"    Type: {prop.value_type.value}")
        if prop.allowed_values:
            print(f"    Values: {', '.join(prop.allowed_values[:5])}...")

    # Test value normalization
    print("\n\nValue Normalization:")
    print("-" * 40)
    tests = [
        ("liita", "pos", "noun"),
        ("liita", "pos", "lila:noun"),
        ("liita", "gender", "masculine"),
        ("parmigiano", "pos", "verb"),
    ]
    for resource, prop, value in tests:
        normalized = registry.normalize_value(resource, prop, value)
        print(f"  {resource}.{prop}({value}) -> {normalized}")

    # Test predicate lookup
    print("\n\nPredicate Lookup:")
    print("-" * 40)
    print(f"  liita.pos -> {registry.get_predicate('liita', 'pos')}")
    print(f"  liita.gender -> {registry.get_predicate('liita', 'gender')}")
    print(f"  liita.part_of_speech -> {registry.get_predicate('liita', 'part_of_speech')}")

    # Test required prefixes
    print("\n\nRequired Prefixes:")
    print("-" * 40)
    prefixes = registry.get_required_prefixes("liita", ["pos", "gender", "written_rep"])
    print(f"  For [pos, gender, written_rep]: {prefixes}")

    print("\n" + "=" * 70)
    print("Property registry ready for integration!")
    print("=" * 70)
