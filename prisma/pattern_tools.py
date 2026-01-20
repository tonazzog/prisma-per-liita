"""
PRISMA Pattern Tools
====================

This module defines pattern-based tools for generating SPARQL query fragments
for the LiITA Knowledge Base. Each tool is a self-contained component that
generates a specific pattern with guaranteed structural correctness.

Part of PRISMA per LiITA - Pattern-based Rules for Intent-driven SPARQL
with Multiple-resource Assembly for the LiITA knowledge base.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from abc import ABC, abstractmethod





# ============================================================================
# Core Data Structures
# ============================================================================

class VariableType(Enum):
    """Types of variables in SPARQL patterns"""
    LEMMA = "lemma"
    WORD = "word"
    SENSE = "sense"
    LEXICAL_ENTRY = "lexical_entry"
    WRITTEN_REP = "written_rep"
    DEFINITION = "definition"
    POS = "pos"


@dataclass
class Variable:
    """Represents a SPARQL variable with type information"""
    name: str  # e.g., "?lemma", "?word"
    type: VariableType
    resource: str  # "liita", "complit", "parmigiano", "sicilian"
    optional: bool = False
    description: str = ""

    def __str__(self):
        return self.name


@dataclass
class PatternFragment:
    """
    A SPARQL pattern fragment with typed inputs/outputs.
    
    This is the fundamental unit of composition. Fragments can be
    assembled together by matching output variables of one fragment
    with input variables of another.
    """
    pattern_name: str
    sparql: str
    input_vars: List[Variable] = field(default_factory=list)
    output_vars: List[Variable] = field(default_factory=list)
    required_prefixes: Set[str] = field(default_factory=set)
    filters_applied: List[str] = field(default_factory=list)
    needs_service_clause: bool = False
    is_optional: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def validate_inputs(self, available_vars: Dict[str, Variable]) -> bool:
        """Check if all required input variables are available"""
        for var in self.input_vars:
            if var.optional:
                continue
            if var.name not in available_vars:
                return False
            # Type compatibility check
            if available_vars[var.name].type != var.type:
                return False
        return True
    
    def get_output_vars_dict(self) -> Dict[str, Variable]:
        """Return output variables as a dictionary for easy lookup"""
        return {var.name: var for var in self.output_vars}


# ============================================================================
# Abstract Base Class for Pattern Tools
# ============================================================================

class PatternTool(ABC):
    """
    Abstract base class for all pattern generation tools.
    
    Each tool is responsible for generating a specific type of SPARQL
    pattern fragment with guaranteed correctness.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this pattern tool"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this pattern does"""
        pass
    
    @property
    def required_prefixes(self) -> Set[str]:
        """SPARQL prefixes required by this pattern"""
        return {
            "lila", "ontolex", "lime", "vartrans", 
            "rdfs", "skos", "lexinfo", "dcterms"
        }
    
    @abstractmethod
    def generate(self, **kwargs) -> PatternFragment:
        """Generate the pattern fragment with given parameters"""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict:
        """Return JSON schema describing required inputs"""
        pass
    
    def validate_params(self, params: Dict) -> tuple[bool, Optional[str]]:
        """Validate input parameters against schema"""
        schema = self.get_input_schema()
        for required_field in schema.get("required", []):
            if required_field not in params:
                return False, f"Missing required parameter: {required_field}"
        return True, None


# ============================================================================
# PATTERN TOOL 1: CompL-it Definition Search
# ============================================================================

class CompLitDefinitionSearchPattern(PatternTool):
    """
    Generates SERVICE clause for searching CompL-it by definition content.
    
    Use Case: "Find words whose definition starts with X"
    
    Template guarantees:
    - Proper SERVICE clause structure
    - Filters inside SERVICE
    - Correct property paths
    - Output variables properly exposed
    """
    
    @property
    def name(self) -> str:
        return "complit_definition_search"
    
    @property
    def description(self) -> str:
        return "Search CompL-it entries by definition pattern"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["definition_pattern"],
            "properties": {
                "definition_pattern": {
                    "type": "string",
                    "description": "Text pattern to match in definitions"
                },
                "pattern_type": {
                    "type": "string",
                    "enum": ["starts_with", "contains", "ends_with", "regex"],
                    "default": "starts_with"
                },
                "pos_filter": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "description": "Optional part-of-speech filter"
                },
                "retrieve_examples": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to retrieve usage examples"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        definition_pattern = kwargs["definition_pattern"]
        pattern_type = kwargs.get("pattern_type", "starts_with")
        pos_filter = kwargs.get("pos_filter")
        retrieve_examples = kwargs.get("retrieve_examples", False)
        
        # Build filter based on pattern type
        filter_map = {
            "starts_with": f'FILTER(strstarts(str(?definition), "{definition_pattern}"))',
            "contains": f'FILTER(contains(str(?definition), "{definition_pattern}"))',
            "ends_with": f'FILTER(strends(str(?definition), "{definition_pattern}"))',
            "regex": f'FILTER(regex(str(?definition), "{definition_pattern}"))'
        }
        definition_filter = filter_map[pattern_type]
        
        # Build POS filter if specified
        pos_clause = ""
        pos_filter_clause = ""
        if pos_filter:
            pos_clause = "lexinfo:partOfSpeech [ rdfs:label ?pos ] ;"
            pos_filter_clause = f'FILTER(str(?pos) = "{pos_filter}") .'
        
        # Build example retrieval if requested
        example_clause = ""
        if retrieve_examples:
            example_clause = """
    OPTIONAL {
      ?sense lexinfo:senseExample ?example
    } ."""
        
        sparql = f"""
  SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {{
    ?word a ontolex:Word ;
          {pos_clause}
          ontolex:sense ?sense ;
          ontolex:canonicalForm ?form .
    ?form ontolex:writtenRep ?lemma .
    OPTIONAL {{
      ?sense skos:definition ?definition
    }} .{example_clause}
    {definition_filter}
    {pos_filter_clause}
  }}"""
        
        # Define output variables
        output_vars = [
            Variable("?word", VariableType.WORD, "complit", 
                    description="CompL-it word entry"),
            Variable("?lemma", VariableType.LEMMA, "complit",
                    description="Written representation of lemma"),
            Variable("?sense", VariableType.SENSE, "complit",
                    description="Lexical sense"),
            Variable("?definition", VariableType.DEFINITION, "complit",
                    optional=True, description="Definition text")
        ]
        
        if retrieve_examples:
            output_vars.append(
                Variable("?example", VariableType.WRITTEN_REP, "complit",
                        optional=True, description="Usage example")
            )
        
        filters = ["definition_pattern"]
        if pos_filter:
            filters.append("pos")
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],  # No inputs - this is a starting pattern
            output_vars=output_vars,
            required_prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
            filters_applied=filters,
            needs_service_clause=True,
            metadata={
                "query_type": "definition_search",
                "pattern": definition_pattern,
                "pattern_type": pattern_type
            }
        )


# ============================================================================
# PATTERN TOOL 2: CompL-it Semantic Relations (Hypernymy/Hyponymy)
# ============================================================================

class CompLitSemanticRelationPattern(PatternTool):
    """
    Generates SERVICE clause for navigating semantic relations in CompL-it.
    
    Use Cases:
    - "Find hyponyms of X" (X is a hypernym of results)
    - "Find hypernyms of X" (X is a hyponym of results)
    - "Find meronyms of X" (X contains results as parts)
    
    Template guarantees:
    - Correct relation property usage
    - Proper sense navigation
    - Related word retrieval
    - Definition collection
    """
    
    @property
    def name(self) -> str:
        return "complit_semantic_relation"
    
    @property
    def description(self) -> str:
        return "Navigate semantic relations (hypernymy, meronymy, etc.) in CompL-it"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["lemma", "relation_type"],
            "properties": {
                "lemma": {
                    "type": "string",
                    "description": "The source lemma to start from"
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["hyponym", "hypernym", "meronym", "holonym", "synonym"],
                    "description": "Type of semantic relation to navigate"
                },
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "default": "noun",
                    "description": "Part of speech of source lemma"
                },
                "retrieve_definitions": {
                    "type": "boolean",
                    "default": True,
                    "description": "Retrieve definitions of related words"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        lemma = kwargs["lemma"]
        relation_type = kwargs["relation_type"]
        pos = kwargs.get("pos", "noun")
        retrieve_definitions = kwargs.get("retrieve_definitions", True)
        
        # Map relation types to property and direction
        relation_map = {
            "hyponym": ("lexinfo:hypernym", "?sense", "?relatedSense"),  # X hypernym Y = Y hyponym X
            "hypernym": ("lexinfo:hyponym", "?sense", "?relatedSense"),
            "meronym": ("lexinfo:partMeronym", "?relatedSense", "?sense"),  # X has part Y
            "holonym": ("lexinfo:partMeronym", "?sense", "?relatedSense"),
            "synonym": ("lexinfo:approximateSynonym", "?sense", "?relatedSense"),
            "antonym": ("lexinfo:antonym", "?sense", "?relatedSense")  # X antonym Y
        }
        
        property_name, subject, object_var = relation_map[relation_type]
        
        # Build definition retrieval
        definition_clause = ""
        if retrieve_definitions:
            definition_clause = """
    OPTIONAL {
      ?relatedSense skos:definition ?definition
    } ."""
        
        sparql = f"""
  SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {{
    ?word a ontolex:Word ;
          lexinfo:partOfSpeech [ rdfs:label ?pos ] ;
          ontolex:sense ?sense ;
          ontolex:canonicalForm [ ontolex:writtenRep ?lemma ] .
    
    {subject} {property_name} {object_var} .{definition_clause}
    
    FILTER(str(?pos) = "{pos}") .
    FILTER(str(?lemma) = "{lemma}") .
    
    ?relatedWord ontolex:sense ?relatedSense .
  }}"""
        
        output_vars = [
            Variable("?relatedWord", VariableType.WORD, "complit",
                    description=f"CompL-it words that are {relation_type}s"),
            Variable("?relatedSense", VariableType.SENSE, "complit",
                    description=f"Sense representing {relation_type} relation")
        ]
        
        if retrieve_definitions:
            output_vars.append(
                Variable("?definition", VariableType.DEFINITION, "complit",
                        optional=True, description="Definition of related word")
            )
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],
            output_vars=output_vars,
            required_prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
            filters_applied=["lemma", "pos"],
            needs_service_clause=True,
            metadata={
                "query_type": "semantic_relation",
                "source_lemma": lemma,
                "relation": relation_type,
                "pos": pos
            }
        )


# ============================================================================
# PATTERN TOOL 2b: CompL-it Word Sense Lookup
# ============================================================================

class CompLitWordSenseLookupPattern(PatternTool):
    """
    Looks up a word in CompL-it by its written form and retrieves all its senses
    with their definitions.

    Use Case: "Find all senses of the word 'vita'"

    Template guarantees:
    - Proper SERVICE clause structure
    - Complete sense and definition retrieval
    - Optional POS filtering
    """

    @property
    def name(self) -> str:
        return "complit_word_sense_lookup"

    @property
    def description(self) -> str:
        return "Look up a word in CompL-it and retrieve all its senses and definitions"

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["lemma"],
            "properties": {
                "lemma": {
                    "type": "string",
                    "description": "The word to look up (written form)"
                },
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "description": "Optional part-of-speech filter"
                },
                "retrieve_examples": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to retrieve usage examples"
                }
            }
        }

    def generate(self, **kwargs) -> PatternFragment:
        lemma = kwargs["lemma"]
        pos = kwargs.get("pos")
        retrieve_examples = kwargs.get("retrieve_examples", False)

        # Build POS clause if specified
        pos_clause = ""
        pos_filter = ""
        if pos:
            pos_clause = "lexinfo:partOfSpeech [ rdfs:label ?pos ] ;"
            pos_filter = f'\n    FILTER(str(?pos) = "{pos}") .'

        # Build example retrieval if requested
        example_clause = ""
        if retrieve_examples:
            example_clause = """
    OPTIONAL {
      ?sense lexinfo:senseExample ?example
    } ."""

        sparql = f"""
  SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {{
    ?word a ontolex:Word ;
          {pos_clause}
          ontolex:sense ?sense ;
          ontolex:canonicalForm ?form .
    ?form ontolex:writtenRep ?lemma .
    OPTIONAL {{
      ?sense skos:definition ?definition
    }} .{example_clause}
    FILTER(str(?lemma) = "{lemma}") .{pos_filter}
  }}"""

        # Define output variables
        output_vars = [
            Variable("?word", VariableType.WORD, "complit",
                    description="CompL-it word entry"),
            Variable("?lemma", VariableType.LEMMA, "complit",
                    description="Written representation of lemma"),
            Variable("?sense", VariableType.SENSE, "complit",
                    description="Lexical sense"),
            Variable("?definition", VariableType.DEFINITION, "complit",
                    optional=True, description="Definition text")
        ]

        if retrieve_examples:
            output_vars.append(
                Variable("?example", VariableType.WRITTEN_REP, "complit",
                        optional=True, description="Usage example")
            )

        filters = ["lemma"]
        if pos:
            filters.append("pos")

        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],  # No inputs - this is a starting pattern
            output_vars=output_vars,
            required_prefixes={"ontolex", "lexinfo", "rdfs", "skos"},
            filters_applied=filters,
            needs_service_clause=True,
            metadata={
                "query_type": "word_sense_lookup",
                "lemma": lemma,
                "pos": pos
            }
        )


# ============================================================================
# PATTERN TOOL 2c: CompL-it Semantic Relation Between Two Lemmas
# ============================================================================

class CompLitRelationBetweenLemmasPattern(PatternTool):
    """
    Checks if a semantic relation exists between two specific lemmas in CompL-it.

    Use Case: "Are 'cane' and 'animale' related?", "Is 'rosso' a hyponym of 'colore'?"

    Template guarantees:
    - Correct bidirectional or unidirectional relation checking
    - Proper sense-level linking
    - Returns the relation type if found
    """

    @property
    def name(self) -> str:
        return "complit_relation_between_lemmas"

    @property
    def description(self) -> str:
        return "Check if a semantic relation exists between two lemmas in CompL-it"

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["lemma_a", "lemma_b"],
            "properties": {
                "lemma_a": {
                    "type": "string",
                    "description": "First lemma to check"
                },
                "lemma_b": {
                    "type": "string",
                    "description": "Second lemma to check"
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["hyponym", "hypernym", "meronym", "holonym", "synonym", "antonym", "any"],
                    "default": "any",
                    "description": "Specific relation to check, or 'any' for all relations"
                },
                "pos": {
                    "type": "string",
                    "default": "noun",
                    "description": "Part of speech filter"
                }
            }
        }

    def generate(self, **kwargs) -> PatternFragment:
        lemma_a = kwargs["lemma_a"]
        lemma_b = kwargs["lemma_b"]
        relation_type = kwargs.get("relation_type", "any")
        pos = kwargs.get("pos", "noun")

        # Build relation triple based on type
        if relation_type == "any":
            # Check for any semantic relation (UNION of common relations)
            relation_clause = """
    {
      { ?senseA lexinfo:hyponym ?senseB . BIND("hyponym" AS ?relationType) }
      UNION
      { ?senseB lexinfo:hyponym ?senseA . BIND("hypernym" AS ?relationType) }
      UNION
      { ?senseA lexinfo:approximateSynonym ?senseB . BIND("synonym" AS ?relationType) }
      UNION
      { ?senseA lexinfo:antonym ?senseB . BIND("antonym" AS ?relationType) }
      UNION
      { ?senseA lexinfo:partMeronym ?senseB . BIND("meronym" AS ?relationType) }
      UNION
      { ?senseB lexinfo:partMeronym ?senseA . BIND("holonym" AS ?relationType) }
    }"""
        else:
            # Specific relation check
            relation_map = {
                "hyponym": ("lexinfo:hyponym", "?senseA", "?senseB"),
                "hypernym": ("lexinfo:hyponym", "?senseB", "?senseA"),
                "meronym": ("lexinfo:partMeronym", "?senseA", "?senseB"),
                "holonym": ("lexinfo:partMeronym", "?senseB", "?senseA"),
                "synonym": ("lexinfo:approximateSynonym", "?senseA", "?senseB"),
                "antonym": ("lexinfo:antonym", "?senseA", "?senseB")
            }
            prop, subj, obj = relation_map[relation_type]
            relation_clause = f'{subj} {prop} {obj} . BIND("{relation_type}" AS ?relationType)'

        sparql = f"""
  SERVICE <https://klab.ilc.cnr.it/graphdb-compl-it/> {{
    ?wordA a ontolex:Word ;
           lexinfo:partOfSpeech [ rdfs:label ?posA ] ;
           ontolex:canonicalForm [ ontolex:writtenRep ?lemmaA ] ;
           ontolex:sense ?senseA .
    FILTER(str(?posA) = "{pos}") .
    FILTER(str(?lemmaA) = "{lemma_a}") .

    ?wordB a ontolex:Word ;
           lexinfo:partOfSpeech [ rdfs:label ?posB ] ;
           ontolex:canonicalForm [ ontolex:writtenRep ?lemmaB ] ;
           ontolex:sense ?senseB .
    FILTER(str(?posB) = "{pos}") .
    FILTER(str(?lemmaB) = "{lemma_b}") .
{relation_clause}
  }}"""

        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],
            output_vars=[
                Variable("?wordA", VariableType.WORD, "complit",
                        description=f"CompL-it word for '{lemma_a}'"),
                Variable("?wordB", VariableType.WORD, "complit",
                        description=f"CompL-it word for '{lemma_b}'"),
                Variable("?senseA", VariableType.SENSE, "complit",
                        description="Sense of first word"),
                Variable("?senseB", VariableType.SENSE, "complit",
                        description="Sense of second word"),
                Variable("?relationType", VariableType.LITERAL, "complit",
                        description="Type of relation found")
            ],
            required_prefixes={"ontolex", "lexinfo", "rdfs"},
            filters_applied=["lemma_a", "lemma_b", "pos"],
            needs_service_clause=True,
            metadata={
                "query_type": "relation_check",
                "lemma_a": lemma_a,
                "lemma_b": lemma_b,
                "relation_type": relation_type,
                "pos": pos
            }
        )


# ============================================================================
# PATTERN TOOL 3: Bridge Pattern (CompL-it → LiITA)
# ============================================================================

class BridgePattern(PatternTool):
    """
    Bridges CompL-it data to LiITA Lemma Bank.

    This is THE critical linking pattern. It must always be used after
    any CompL-it pattern to connect external data to local resources.

    Template guarantees:
    - Correct use of ontolex:canonicalForm
    - Variable naming consistency
    - Type compatibility
    """
    
    @property
    def name(self) -> str:
        return "bridge_complit_to_liita"
    
    @property
    def description(self) -> str:
        return "Bridge CompL-it word entries to LiITA lemmas"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["source_var"],
            "properties": {
                "source_var": {
                    "type": "string",
                    "description": "CompL-it word variable to bridge (e.g., '?word' or '?relatedWord')"
                },
                "target_var": {
                    "type": "string",
                    "default": "?liitaLemma",
                    "description": "Name for the LiITA lemma variable"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        source_var = kwargs["source_var"]
        target_var = kwargs.get("target_var", "?liitaLemma")
        
        # Ensure variables start with ?
        if not source_var.startswith("?"):
            source_var = "?" + source_var
        if not target_var.startswith("?"):
            target_var = "?" + target_var
        
        sparql = f"""
  {source_var} ontolex:canonicalForm {target_var} ."""
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(source_var, VariableType.WORD, "complit",
                        description="CompL-it word to bridge from")
            ],
            output_vars=[
                Variable(target_var, VariableType.LEMMA, "liita",
                        description="LiITA lemma (bridge point)")
            ],
            required_prefixes={"ontolex"},
            filters_applied=[],
            needs_service_clause=False,
            metadata={
                "bridge_type": "complit_to_liita",
                "connects": [source_var, target_var]
            }
        )


# ============================================================================
# PATTERN TOOL 4: Parmigiano Translation Link
# ============================================================================

class ParmigianoTranslationPattern(PatternTool):
    """
    Links Italian LiITA lemmas to Parmigiano dialect translations.
    
    Use Case: "Show Parmigiano translation of X"
    
    Template guarantees:
    - Correct lexicon URI
    - Proper use of vartrans:translatableAs
    - Correct direction (Italian → Parmigiano)
    - Written representation retrieval
    """
    
    @property
    def name(self) -> str:
        return "parmigiano_translation"
    
    @property
    def description(self) -> str:
        return "Link Italian lemmas to Parmigiano dialect translations"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["italian_lemma_var"],
            "properties": {
                "italian_lemma_var": {
                    "type": "string",
                    "description": "Variable holding Italian LiITA lemma (e.g., '?liitaLemma')"
                },
                "parmigiano_var_prefix": {
                    "type": "string",
                    "default": "parmigiano",
                    "description": "Prefix for generated Parmigiano variables"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        italian_var = kwargs["italian_lemma_var"]
        prefix = kwargs.get("parmigiano_var_prefix", "parmigiano")
        
        if not italian_var.startswith("?"):
            italian_var = "?" + italian_var
        
        # Generate consistent variable names
        le_ita_var = f"?le{prefix}Ita"
        le_par_var = f"?le{prefix}Par"
        par_lemma_var = f"?{prefix}Lemma"
        par_wr_var = f"?{prefix}WR"
        
        sparql = f"""
  {le_ita_var} ontolex:canonicalForm {italian_var} ;
               ^lime:entry <http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon> .
  {le_ita_var} vartrans:translatableAs {le_par_var} .
  {le_par_var} ontolex:canonicalForm {par_lemma_var} .
  {par_lemma_var} ontolex:writtenRep {par_wr_var} ."""
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(italian_var, VariableType.LEMMA, "liita",
                        description="Italian LiITA lemma")
            ],
            output_vars=[
                Variable(par_lemma_var, VariableType.LEMMA, "parmigiano",
                        description="Parmigiano lemma"),
                Variable(par_wr_var, VariableType.WRITTEN_REP, "parmigiano",
                        description="Parmigiano written form")
            ],
            required_prefixes={"ontolex", "lime", "vartrans"},
            filters_applied=[],
            needs_service_clause=False,
            metadata={
                "dialect": "parmigiano",
                "lexicon_uri": "http://liita.it/data/id/LexicalReources/DialettoParmigiano/Lexicon"
            }
        )


# ============================================================================
# PATTERN TOOL 5: Sicilian Translation Link
# ============================================================================

class SicilianTranslationPattern(PatternTool):
    """
    Links Italian LiITA lemmas to Sicilian dialect translations.
    
    Similar to Parmigiano but uses different lexicon URI.
    """
    
    @property
    def name(self) -> str:
        return "sicilian_translation"
    
    @property
    def description(self) -> str:
        return "Link Italian lemmas to Sicilian dialect translations"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["italian_lemma_var"],
            "properties": {
                "italian_lemma_var": {
                    "type": "string",
                    "description": "Variable holding Italian LiITA lemma"
                },
                "sicilian_var_prefix": {
                    "type": "string",
                    "default": "sicilian",
                    "description": "Prefix for generated Sicilian variables"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        italian_var = kwargs["italian_lemma_var"]
        prefix = kwargs.get("sicilian_var_prefix", "sicilian")

        if not italian_var.startswith("?"):
            italian_var = "?" + italian_var

        le_ita_var = f"?le{prefix}Ita"
        le_sic_var = f"?le{prefix}Sic"
        sic_lemma_var = f"?{prefix}Lemma"
        sic_wr_var = f"?{prefix}WR"

        # Filter to Sicilian lexicon using dcterms:isPartOf on the lemma
        sparql = f"""
  {le_sic_var} ontolex:canonicalForm {sic_lemma_var} .
  {le_ita_var} vartrans:translatableAs {le_sic_var} ;
               ontolex:canonicalForm {italian_var} .
  {sic_lemma_var} dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;
                  ontolex:writtenRep {sic_wr_var} ."""

        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(italian_var, VariableType.LEMMA, "liita",
                        description="Italian LiITA lemma")
            ],
            output_vars=[
                Variable(sic_lemma_var, VariableType.LEMMA, "sicilian",
                        description="Sicilian lemma"),
                Variable(sic_wr_var, VariableType.WRITTEN_REP, "sicilian",
                        description="Sicilian written form")
            ],
            required_prefixes={"ontolex", "vartrans", "dcterms"},
            filters_applied=["sicilian_lexicon"],
            needs_service_clause=False,
            metadata={
                "dialect": "sicilian"
            }
        )


# ============================================================================
# PATTERN TOOL 6: Sicilian Pattern Search (No Italian Required)
# ============================================================================

class SicilianPatternSearchPattern(PatternTool):
    """
    Searches Sicilian lemmas by written form pattern, then optionally
    links to Italian translations.
    
    Use Case: "Find Sicilian words ending in 'ìa'"
    
    This is different from SicilianTranslationPattern because it starts
    from Sicilian rather than Italian.
    """
    
    @property
    def name(self) -> str:
        return "sicilian_pattern_search"
    
    @property
    def description(self) -> str:
        return "Search Sicilian lemmas by pattern, optionally link to Italian"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern for Sicilian written forms"
                },
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "description": "Optional POS filter"
                },
                "link_to_italian": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to retrieve Italian translations"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        pattern = kwargs["pattern"]
        pos = kwargs.get("pos")
        link_to_italian = kwargs.get("link_to_italian", True)
        
        # POS filter clause
        pos_clause = ""
        pos_filter = ""
        if pos:
            pos_clause = "\n                 lila:hasPOS ?pos ."
            pos_filter = f"\n  FILTER(?pos = lila:{pos}) ."

        # Italian linking clause
        italian_clause = ""
        if link_to_italian:
            italian_clause = """
  ?leSicilian ontolex:canonicalForm ?sicilianLemma .
  ?leItalian vartrans:translatableAs ?leSicilian ;
             ontolex:canonicalForm ?liitaLemma .
  ?liitaLemma ontolex:writtenRep ?italianWR ."""

        # Build the base triple pattern - end with . if no pos_clause, else with ;
        if pos:
            sparql = f"""
  ?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?sicilianWR ;{pos_clause}
  FILTER(regex(str(?sicilianWR), "{pattern}")) .{pos_filter}{italian_clause}"""
        else:
            sparql = f"""
  ?sicilianLemma dcterms:isPartOf <http://liita.it/data/id/DialettoSiciliano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?sicilianWR .
  FILTER(regex(str(?sicilianWR), "{pattern}")) .{italian_clause}"""
        
        output_vars = [
            Variable("?sicilianLemma", VariableType.LEMMA, "sicilian",
                    description="Sicilian lemma matching pattern"),
            Variable("?sicilianWR", VariableType.WRITTEN_REP, "sicilian",
                    description="Sicilian written representation")
        ]
        
        if link_to_italian:
            output_vars.extend([
                Variable("?liitaLemma", VariableType.LEMMA, "liita",
                        description="Italian translation lemma"),
                Variable("?italianWR", VariableType.WRITTEN_REP, "liita",
                        description="Italian written representation")
            ])
        
        filters = ["sicilian_pattern"]
        if pos:
            filters.append("pos")
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],
            output_vars=output_vars,
            required_prefixes={"ontolex", "dcterms", "lila", "vartrans"},
            filters_applied=filters,
            needs_service_clause=False,
            metadata={
                "query_type": "pattern_search",
                "dialect": "sicilian",
                "pattern": pattern
            }
        )


# ============================================================================
# PATTERN TOOL 7: LiITA Basic Query
# ============================================================================

class LiITABasicQueryPattern(PatternTool):
    """
    Queries LiITA Lemma Bank directly without external resources.
    
    Use Cases:
    - "How many nouns in LiITA?"
    - "Find lemmas starting with 'infra'"
    - "List all verbs"
    """
    
    @property
    def name(self) -> str:
        return "liita_basic_query"
    
    @property
    def description(self) -> str:
        return "Query LiITA Lemma Bank for lemmas with filters"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "pos_filter": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb", "pronoun", "determiner"],
                    "description": "Filter by part of speech"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern for written representation"
                },
                "pattern_type": {
                    "type": "string",
                    "enum": ["starts_with", "ends_with", "contains", "regex"],
                    "default": "regex"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        pos_filter = kwargs.get("pos_filter")
        pattern = kwargs.get("pattern")
        pattern_type = kwargs.get("pattern_type", "regex")
        
        # Build POS clause
        pos_clause = ""
        if pos_filter:
            pos_clause = f"?lemma lila:hasPOS lila:{pos_filter} ."
        
        # Build pattern filter
        pattern_filter = ""
        if pattern:
            if pattern_type == "starts_with":
                pattern_filter = f'FILTER(regex(str(?wr), "^{pattern}")) .'
            elif pattern_type == "ends_with":
                pattern_filter = f'FILTER(regex(str(?wr), "{pattern}$")) .'
            elif pattern_type == "contains":
                pattern_filter = f'FILTER(contains(str(?wr), "{pattern}")) .'
            else:  # regex
                pattern_filter = f'FILTER(regex(str(?wr), "{pattern}")) .'
        
        sparql = f"""
  GRAPH <http://liita.it/data> {{
    ?lemma a lila:Lemma .
    {pos_clause}
    ?lemma ontolex:writtenRep ?wr .
    {pattern_filter}
  }}"""
        
        filters = []
        if pos_filter:
            filters.append("pos")
        if pattern:
            filters.append("pattern")
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],
            output_vars=[
                Variable("?lemma", VariableType.LEMMA, "liita",
                        description="LiITA lemma"),
                Variable("?wr", VariableType.WRITTEN_REP, "liita",
                        description="Written representation")
            ],
            required_prefixes={"lila", "ontolex"},
            filters_applied=filters,
            needs_service_clause=False,
            metadata={
                "query_type": "basic_liita",
                "pos": pos_filter,
                "pattern": pattern
            }
        )


# ============================================================================
# PATTERN TOOL 8: Aggregation Pattern
# ============================================================================

class AggregationPattern(PatternTool):
    """
    Handles aggregation operations: COUNT, GROUP_CONCAT, etc.
    
    This is a special pattern that wraps around SELECT clause rather
    than generating WHERE patterns.
    """
    
    @property
    def name(self) -> str:
        return "aggregation"
    
    @property
    def description(self) -> str:
        return "Add aggregation functions to SELECT clause"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["aggregation_type"],
            "properties": {
                "aggregation_type": {
                    "type": "string",
                    "enum": ["count", "group_concat", "distinct"],
                    "description": "Type of aggregation"
                },
                "aggregate_var": {
                    "type": "string",
                    "description": "Variable to aggregate (for COUNT, GROUP_CONCAT)"
                },
                "group_by_vars": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables to group by"
                },
                "separator": {
                    "type": "string",
                    "default": ", ",
                    "description": "Separator for GROUP_CONCAT"
                },
                "order_by": {
                    "type": "object",
                    "properties": {
                        "var": {"type": "string"},
                        "direction": {"enum": ["ASC", "DESC"]}
                    }
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        agg_type = kwargs["aggregation_type"]
        aggregate_var = kwargs.get("aggregate_var")
        group_by_vars = kwargs.get("group_by_vars", [])
        separator = kwargs.get("separator", ", ")
        order_by = kwargs.get("order_by")
        
        # This pattern doesn't generate WHERE clauses, but SELECT modifiers
        # It returns metadata for the query builder
        
        metadata = {
            "type": agg_type,
            "aggregate_var": aggregate_var,
            "group_by": group_by_vars,
            "separator": separator,
            "order_by": order_by
        }
        
        # Generate SELECT clause modification
        select_clause = ""
        if agg_type == "count":
            select_clause = f"(COUNT(*) as ?count)"
        elif agg_type == "group_concat":
            if not aggregate_var.startswith("?"):
                aggregate_var = "?" + aggregate_var
            select_clause = f"(GROUP_CONCAT(str({aggregate_var}); SEPARATOR=\"{separator}\") AS ?aggregated)"
        elif agg_type == "distinct":
            select_clause = "DISTINCT"
        
        group_by_clause = ""
        if group_by_vars:
            vars_str = " ".join([v if v.startswith("?") else "?" + v for v in group_by_vars])
            group_by_clause = f"\nGROUP BY {vars_str}"
        
        order_by_clause = ""
        if order_by:
            var = order_by["var"]
            if not var.startswith("?"):
                var = "?" + var
            direction = order_by.get("direction", "ASC")
            order_by_clause = f"\nORDER BY {direction}({var})"
        
        return PatternFragment(
            pattern_name=self.name,
            sparql="",  # No WHERE pattern
            input_vars=[],
            output_vars=[],
            required_prefixes=set(),
            filters_applied=[],
            needs_service_clause=False,
            metadata={
                "is_aggregation": True,
                "select_clause": select_clause,
                "group_by_clause": group_by_clause,
                "order_by_clause": order_by_clause,
                **metadata
            }
        )


# ============================================================================
# PATTERN TOOL 9: Optional Italian Written Representation
# ============================================================================

class ItalianWrittenRepPattern(PatternTool):
    """
    Retrieves Italian written representation for a LiITA lemma.
    
    Often needed when you have a lemma URI but want to display the actual word.
    """
    
    @property
    def name(self) -> str:
        return "italian_written_rep"
    
    @property
    def description(self) -> str:
        return "Get Italian written representation from LiITA lemma"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["lemma_var"],
            "properties": {
                "lemma_var": {
                    "type": "string",
                    "description": "LiITA lemma variable"
                },
                "output_var": {
                    "type": "string",
                    "default": "?italianWR",
                    "description": "Variable name for written representation"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        lemma_var = kwargs["lemma_var"]
        output_var = kwargs.get("output_var", "?italianWR")
        
        if not lemma_var.startswith("?"):
            lemma_var = "?" + lemma_var
        if not output_var.startswith("?"):
            output_var = "?" + output_var
        
        sparql = f"""
  {lemma_var} ontolex:writtenRep {output_var} ."""
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(lemma_var, VariableType.LEMMA, "liita",
                        description="LiITA lemma")
            ],
            output_vars=[
                Variable(output_var, VariableType.WRITTEN_REP, "liita",
                        description="Italian written form")
            ],
            required_prefixes={"ontolex"},
            filters_applied=[],
            needs_service_clause=False,
            metadata={}
        )

# ============================================================================
# PATTERN TOOL 10: Parmigiano Pattern Search
# ============================================================================

class ParmigianoPatternSearchPattern(PatternTool):
    """
    Searches Parmigiano lemmas by written form pattern, then optionally
    links to Italian translations.
    
    Use Case: "Find Parmigiano words ending in 'ìa'"
    
    """
    
    @property
    def name(self) -> str:
        return "parmigiano_pattern_search"
    
    @property
    def description(self) -> str:
        return "Search Parmigiano lemmas by pattern, optionally link to Italian"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern for Parmigiano written forms"
                },
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "description": "Optional POS filter"
                },
                "link_to_italian": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to retrieve Italian translations"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        pattern = kwargs["pattern"]
        pos = kwargs.get("pos")
        link_to_italian = kwargs.get("link_to_italian", True)
        
        # POS filter clause
        pos_clause = ""
        pos_filter = ""
        if pos:
            pos_clause = "\n                 lila:hasPOS ?pos ."
            pos_filter = f"\n  FILTER(?pos = lila:{pos}) ."

        # Italian linking clause
        italian_clause = ""
        if link_to_italian:
            italian_clause = """
  ?leParmigiano ontolex:canonicalForm ?parmigianoLemma .
  ?leItalian vartrans:translatableAs ?leParmigiano ;
             ontolex:canonicalForm ?liitaLemma .
  ?liitaLemma ontolex:writtenRep ?italianWR ."""

        # Build the base triple pattern - end with . if no pos_clause, else with ;
        if pos:
            sparql = f"""
  ?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?parmigianoWR ;{pos_clause}
  FILTER(regex(str(?parmigianoWR), "{pattern}")) .{pos_filter}{italian_clause}"""
        else:
            sparql = f"""
  ?parmigianoLemma dcterms:isPartOf <http://liita.it/data/id/DialettoParmigiano/lemma/LemmaBank> ;
                 ontolex:writtenRep ?parmigianoWR .
  FILTER(regex(str(?parmigianoWR), "{pattern}")) .{italian_clause}"""
        
        output_vars = [
            Variable("?parmigianoLemma", VariableType.LEMMA, "parmigiano",
                    description="Parmigiano lemma matching pattern"),
            Variable("?parmigianoWR", VariableType.WRITTEN_REP, "parmigiano",
                    description="Parmigiano written representation")
        ]
        
        if link_to_italian:
            output_vars.extend([
                Variable("?liitaLemma", VariableType.LEMMA, "liita",
                        description="Italian translation lemma"),
                Variable("?italianWR", VariableType.WRITTEN_REP, "liita",
                        description="Italian written representation")
            ])
        
        filters = ["parmigiano_pattern"]
        if pos:
            filters.append("pos")
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[],
            output_vars=output_vars,
            required_prefixes={"ontolex", "dcterms", "lila", "vartrans"},
            filters_applied=filters,
            needs_service_clause=False,
            metadata={
                "query_type": "pattern_search",
                "dialect": "parmigiano",
                "pattern": pattern
            }
        )

    
# ============================================================================
# PATTERN TOOL 11: Sentix Linking Pattern
# ============================================================================

class SentixLinkingPattern(PatternTool):
    """
    Links LiITA lemmas to Sentix affective lexicon entries.
    
    Use Case: "What is the polarity of X?", "Show Sentix sentiment for Y"
    
    Based on the paper's SPARQL queries, Sentix uses:
    - marl:hasPolarity for polarity type (Positive/Negative/Neutral)
    - marl:hasPolarityValue for numeric score (-1 to +1)
    
    Linking via ontolex:canonicalForm (simpler than dialect patterns).
    """
    
    @property
    def name(self) -> str:
        return "sentix_linking"
    
    @property
    def description(self) -> str:
        return "Link LiITA lemmas to Sentix affective lexicon entries"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["liita_lemma_var"],
            "properties": {
                "liita_lemma_var": {
                    "type": "string",
                    "description": "Variable holding LiITA lemma (e.g., '?lemma')"
                },
                "retrieve_polarity_type": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to retrieve polarity type (Positive/Negative/Neutral)"
                },
                "retrieve_polarity_value": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to retrieve numeric polarity value (-1 to +1)"
                },
                "polarity_filter": {
                    "type": "string",
                    "enum": ["Positive", "Negative", "Neutral"],
                    "description": "Optional filter by polarity type"
                },
                "polarity_value_min": {
                    "type": "number",
                    "description": "Optional minimum polarity value filter"
                },
                "polarity_value_max": {
                    "type": "number",
                    "description": "Optional maximum polarity value filter"
                }
            }
        }
    
    def generate(self, **kwargs) -> PatternFragment:
        liita_var = kwargs["liita_lemma_var"]
        retrieve_polarity_type = kwargs.get("retrieve_polarity_type", True)
        retrieve_polarity_value = kwargs.get("retrieve_polarity_value", True)
        polarity_filter = kwargs.get("polarity_filter")
        polarity_value_min = kwargs.get("polarity_value_min")
        polarity_value_max = kwargs.get("polarity_value_max")
        
        if not liita_var.startswith("?"):
            liita_var = "?" + liita_var
        
        # Build SPARQL pattern
        sparql_parts = [
            f"  ?sentixLemma ontolex:canonicalForm {liita_var} ."
        ]
        
        # Add polarity type retrieval
        if retrieve_polarity_type:
            sparql_parts.append("  ?sentixLemma marl:hasPolarity ?polarity .")
            sparql_parts.append("  ?polarity rdfs:label ?polarityLabel .")
        
        # Add polarity value retrieval
        if retrieve_polarity_value:
            sparql_parts.append("  ?sentixLemma marl:hasPolarityValue ?polarityValue .")
        
        # Add filters
        filters = []
        if polarity_filter:
            sparql_parts.append(f'  FILTER(?polarityLabel = "{polarity_filter}"@en) .')
            filters.append("polarity_type")
        
        if polarity_value_min is not None:
            sparql_parts.append(f"  FILTER(?polarityValue >= {polarity_value_min}) .")
            filters.append("polarity_value_min")
        
        if polarity_value_max is not None:
            sparql_parts.append(f"  FILTER(?polarityValue <= {polarity_value_max}) .")
            filters.append("polarity_value_max")
        
        sparql = "\n".join(sparql_parts)
        
        # Define output variables
        output_vars = [
            Variable("?sentixLemma", VariableType.LEMMA, "sentix",
                    description="Sentix lemma entry")
        ]
        
        if retrieve_polarity_type:
            output_vars.extend([
                Variable("?polarity", VariableType.WRITTEN_REP, "sentix",
                        description="Polarity instance"),
                Variable("?polarityLabel", VariableType.WRITTEN_REP, "sentix",
                        description="Polarity label (Positive/Negative/Neutral)")
            ])
        
        if retrieve_polarity_value:
            output_vars.append(
                Variable("?polarityValue", VariableType.WRITTEN_REP, "sentix",
                        description="Polarity numeric value (-1 to +1)")
            )
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(liita_var, VariableType.LEMMA, "liita",
                        description="LiITA lemma")
            ],
            output_vars=output_vars,
            required_prefixes={"ontolex", "marl", "rdfs"},
            filters_applied=filters,
            needs_service_clause=False,
            metadata={
                "resource": "sentix",
                "retrieve_polarity_type": retrieve_polarity_type,
                "retrieve_polarity_value": retrieve_polarity_value
            }
        )


# ============================================================================
# Emotion Keyword Mapping (for ELIta)
# ============================================================================

EMOTION_MAP = {
    # Italian emotion keywords -> ELIta IRI
    "gioia": "elita:Gioia",
    "tristezza": "elita:Tristezza",
    "paura": "elita:Paura",
    "rabbia": "elita:Rabbia",
    "disgusto": "elita:Disgusto",
    "sorpresa": "elita:Sorpresa",
    "aspettativa": "elita:Aspettativa",
    "fiducia": "elita:Fiducia",
    "amore": "elita:Amore",
    # English emotion keywords -> ELIta IRI
    "joy": "elita:Gioia",
    "happiness": "elita:Gioia",
    "sadness": "elita:Tristezza",
    "fear": "elita:Paura",
    "anger": "elita:Rabbia",
    "disgust": "elita:Disgusto",
    "surprise": "elita:Sorpresa",
    "anticipation": "elita:Aspettativa",
    "trust": "elita:Fiducia",
    "love": "elita:Amore",
}


def resolve_emotion_iris(emotion_terms: List[str]) -> List[str]:
    """
    Resolve emotion keywords to ELIta IRIs.

    Args:
        emotion_terms: List of emotion keywords (Italian or English)

    Returns:
        List of ELIta IRIs (e.g., ["elita:Gioia", "elita:Tristezza"])
    """
    iris = []
    for term in emotion_terms:
        term_lower = term.lower().strip()
        if term_lower in EMOTION_MAP:
            iri = EMOTION_MAP[term_lower]
            if iri not in iris:
                iris.append(iri)
    return iris


# ============================================================================
# PATTERN TOOL 12: ELIta Linking Pattern
# ============================================================================

class ELItaLinkingPattern(PatternTool):
    """
    Links LiITA lemmas to ELIta emotion lexicon entries.

    Use Case: "What emotions are associated with X?", "Show ELIta emotions for Y"
    
    Based on the paper's SPARQL queries, ELIta uses:
    - elita:HasEmotion for emotion instances (elita:Gioia, elita:Tristezza, etc.)
    - Emotions have rdfs:label for readable names
    
    Linking via ontolex:canonicalForm (simpler than dialect patterns).
    """
    
    @property
    def name(self) -> str:
        return "elita_linking"
    
    @property
    def description(self) -> str:
        return "Link LiITA lemmas to ELIta emotion lexicon entries"
    
    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["liita_lemma_var"],
            "properties": {
                "liita_lemma_var": {
                    "type": "string",
                    "description": "Variable holding LiITA lemma (e.g., '?lemma')"
                },
                "emotion_filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of emotion keywords to filter (e.g., ['joy', 'sadness'] or ['Gioia', 'Tristezza'])"
                },
                "retrieve_emotion_label": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to retrieve emotion labels"
                }
            }
        }

    def generate(self, **kwargs) -> PatternFragment:
        liita_var = kwargs["liita_lemma_var"]
        emotion_filters = kwargs.get("emotion_filters", [])
        retrieve_emotion_label = kwargs.get("retrieve_emotion_label", True)

        if not liita_var.startswith("?"):
            liita_var = "?" + liita_var

        # Build SPARQL pattern
        sparql_parts = []

        # Resolve emotion keywords to ELIta IRIs using EMOTION_MAP
        emotion_iris = []
        label_filters = []
        if emotion_filters:
            for term in emotion_filters:
                term_lower = term.lower().strip()
                if term_lower in EMOTION_MAP:
                    iri = EMOTION_MAP[term_lower]
                    if iri not in emotion_iris:
                        emotion_iris.append(iri)
                else:
                    # Not in map - might be a direct ELIta name like "Gioia"
                    # Try prefixing with elita:
                    emotion_iris.append(f"elita:{term}")

        # If we have emotion IRIs, use VALUES clause (cleaner and more efficient)
        filters = []
        if emotion_iris:
            values_str = " ".join(emotion_iris)
            sparql_parts.append(f"  VALUES ?emotion {{ {values_str} }}")
            filters.append("emotion_type")

        sparql_parts.extend([
            f"  ?elitaLemma ontolex:canonicalForm {liita_var} .",
            "  ?elitaLemma elita:HasEmotion ?emotion ."
        ])

        # Add emotion label retrieval
        if retrieve_emotion_label:
            sparql_parts.append("  ?emotion rdfs:label ?emotionLabel .")
        
        sparql = "\n".join(sparql_parts)
        
        # Define output variables
        output_vars = [
            Variable("?elitaLemma", VariableType.LEMMA, "elita",
                    description="ELIta lemma entry"),
            Variable("?emotion", VariableType.WRITTEN_REP, "elita",
                    description="Emotion instance (e.g., elita:Gioia)")
        ]
        
        if retrieve_emotion_label:
            output_vars.append(
                Variable("?emotionLabel", VariableType.WRITTEN_REP, "elita",
                        description="Emotion label (e.g., 'Gioia', 'Tristezza')")
            )
        
        return PatternFragment(
            pattern_name=self.name,
            sparql=sparql,
            input_vars=[
                Variable(liita_var, VariableType.LEMMA, "liita",
                        description="LiITA lemma")
            ],
            output_vars=output_vars,
            required_prefixes={"ontolex", "elita", "rdfs"},
            filters_applied=filters,
            needs_service_clause=False,
            metadata={
                "resource": "elita",
                "emotion_filters": emotion_filters
            }
        )


# ============================================================================
# Pattern Tool Registry
# ============================================================================

class PatternToolRegistry:
    """
    Central registry for all pattern tools.
    Provides lookup and discovery functionality.
    """
    
    def __init__(self):
        self._tools: Dict[str, PatternTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all built-in pattern tools"""
        tools = [
            CompLitDefinitionSearchPattern(),
            CompLitSemanticRelationPattern(),
            CompLitWordSenseLookupPattern(),
            CompLitRelationBetweenLemmasPattern(),
            BridgePattern(),
            ParmigianoTranslationPattern(),
            ParmigianoPatternSearchPattern(),
            SicilianTranslationPattern(),
            SicilianPatternSearchPattern(),
            LiITABasicQueryPattern(),
            AggregationPattern(),
            ItalianWrittenRepPattern(),
            SentixLinkingPattern(),
            ELItaLinkingPattern()
        ]
        
        for tool in tools:
            self.register(tool)
    
    def register(self, tool: PatternTool):
        """Register a pattern tool"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[PatternTool]:
        """Get a pattern tool by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())
    
    def get_tools_by_category(self, category: str) -> List[PatternTool]:
        """Get tools by category (complit, dialect, basic, etc.)"""
        category_map = {
            "complit": ["complit_definition_search", "complit_semantic_relation", "complit_word_sense_lookup", "complit_relation_between_lemmas"],
            "bridge": ["bridge_complit_to_liita"],
            "dialect": [
                "parmigiano_translation", 
                "parmigiano_pattern_search",
                "sicilian_translation", 
                "sicilian_pattern_search"
            ],
            "affective": [
                "sentix_linking",
                "elita_linking"
            ],
            "basic": ["liita_basic_query", "italian_written_rep"],
            "aggregation": ["aggregation"]
        }
        
        tool_names = category_map.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example 1: Generate CompL-it definition search pattern
    print("=" * 70)
    print("EXAMPLE 1: CompL-it Definition Search")
    print("=" * 70)
    
    tool = CompLitDefinitionSearchPattern()
    fragment = tool.generate(
        definition_pattern="uccello",
        pattern_type="starts_with",
        pos_filter="noun"
    )
    
    print(f"Pattern Name: {fragment.pattern_name}")
    print(f"Needs SERVICE: {fragment.needs_service_clause}")
    print(f"\nGenerated SPARQL Fragment:")
    print(fragment.sparql)
    print(f"\nOutput Variables:")
    for var in fragment.output_vars:
        print(f"  - {var.name} ({var.type.value}): {var.description}")
    
    # Example 2: Generate semantic relation pattern
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CompL-it Semantic Relations (Hyponyms)")
    print("=" * 70)
    
    tool = CompLitSemanticRelationPattern()
    fragment = tool.generate(
        lemma="colore",
        relation_type="hyponym",
        pos="noun"
    )
    
    print(f"Pattern Name: {fragment.pattern_name}")
    print(f"\nGenerated SPARQL Fragment:")
    print(fragment.sparql)
    
    # Example 3: Generate bridge pattern
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Bridge Pattern (CompL-it → LiITA)")
    print("=" * 70)
    
    tool = BridgePattern()
    fragment = tool.generate(source_var="?relatedWord")
    
    print(f"Pattern Name: {fragment.pattern_name}")
    print(f"Input Variables: {[str(v) for v in fragment.input_vars]}")
    print(f"Output Variables: {[str(v) for v in fragment.output_vars]}")
    print(f"\nGenerated SPARQL Fragment:")
    print(fragment.sparql)
    
    # Example 4: Generate Parmigiano translation pattern
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Parmigiano Translation")
    print("=" * 70)
    
    tool = ParmigianoTranslationPattern()
    fragment = tool.generate(italian_lemma_var="?liitaLemma")
    
    print(f"Pattern Name: {fragment.pattern_name}")
    print(f"\nGenerated SPARQL Fragment:")
    print(fragment.sparql)
    
    # Example 5: Complete pattern chain
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Pattern Chain")
    print("Query: 'Find hyponyms of colore with Parmigiano translations'")
    print("=" * 70)
    
    registry = PatternToolRegistry()
    
    # Step 1: CompL-it semantic query
    semantic_tool = registry.get("complit_semantic_relation")
    semantic_frag = semantic_tool.generate(
        lemma="colore",
        relation_type="hyponym",
        pos="noun"
    )
    
    # Step 2: Bridge
    bridge_tool = registry.get("bridge_complit_to_liita")
    bridge_frag = bridge_tool.generate(source_var="?relatedWord")
    
    # Step 3: Parmigiano link
    parm_tool = registry.get("parmigiano_translation")
    parm_frag = parm_tool.generate(italian_lemma_var="?liitaLemma")
    
    # Step 4: Aggregation
    agg_tool = registry.get("aggregation")
    agg_frag = agg_tool.generate(
        aggregation_type="group_concat",
        aggregate_var="?definition",
        group_by_vars=["relatedSense", "liitaLemma", "parmigianoLemma", "parmigianoWR"],
        separator="; ",
        order_by={"var": "parmigianoWR", "direction": "ASC"}
    )
    
    print("\nPattern Chain:")
    print(f"1. {semantic_frag.pattern_name}")
    print(f"2. {bridge_frag.pattern_name}")
    print(f"3. {parm_frag.pattern_name}")
    print(f"4. {agg_frag.pattern_name}")
    
    print("\nVariable Flow:")
    print(f"SERVICE: outputs {[str(v) for v in semantic_frag.output_vars]}")
    print(f"BRIDGE: takes {[str(v) for v in bridge_frag.input_vars]}, outputs {[str(v) for v in bridge_frag.output_vars]}")
    print(f"PARMIGIANO: takes {[str(v) for v in parm_frag.input_vars]}, outputs {[str(v) for v in parm_frag.output_vars]}")
    
    print("\n" + "=" * 70)
    print("Pattern Tool Registry")
    print("=" * 70)
    print(f"Total tools registered: {len(registry.list_tools())}")
    print("\nTools by category:")
    for category in ["complit", "bridge", "dialect", "basic", "aggregation"]:
        tools = registry.get_tools_by_category(category)
        print(f"\n{category.upper()}:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")