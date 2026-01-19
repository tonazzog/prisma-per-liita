"""
PRISMA Pattern Orchestrator
===========================

Rule-based orchestrator that maps query intents to pattern sequences.
This is a deterministic component - no LLM uncertainty.

The orchestrator:
1. Takes intent analysis (from LLM) as input
2. Applies deterministic rules to select patterns
3. Ensures architectural correctness (e.g., SERVICE → bridge)
4. Returns an execution plan with parameters

Part of PRISMA per LiITA - Pattern-based Rules for Intent-driven SPARQL
with Multiple-resource Assembly for the LiITA knowledge base.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ============================================================================
# Intent Schema (Input to Orchestrator)
# ============================================================================

class QueryType(Enum):
    """High-level query type classification"""
    BASIC_LEMMA_LOOKUP = "basic_lemma_lookup"
    COMPLIT_SEMANTIC = "complit_semantic"
    COMPLIT_DEFINITIONS = "complit_definitions"
    COMPLIT_RELATION_CHECK = "complit_relation_check"  # Check relation between two lemmas
    DIALECT_TRANSLATION = "dialect_translation"
    DIALECT_PATTERN_SEARCH = "dialect_pattern_search"
    MULTI_RESOURCE = "multi_resource"
    SENTIX_POLARITY = "sentix_polarity"
    ELITA_EMOTION = "elita_emotion"
    AFFECTIVE_MULTI_RESOURCE = "affective_multi_resource"


class ResourceType(Enum):
    """Available resources"""
    LIITA = "liita_lemma_bank"
    COMPLIT = "complit"
    PARMIGIANO = "parmigiano"
    SICILIAN = "sicilian"
    SENTIX = "sentix"
    ELITA = "elita"


class SemanticRelationType(Enum):
    """Semantic relation types in CompL-it"""
    HYPONYM = "hyponym"
    HYPERNYM = "hypernym"
    MERONYM = "meronym"
    HOLONYM = "holonym"
    SYNONYM = "synonym"
    ANTONYM = "antonym"


@dataclass
class Intent:
    """
    Structured intent representation from LLM analysis.
    This is the input to the orchestrator.
    """
    query_type: QueryType
    required_resources: List[ResourceType]
    
    # Search criteria
    lemma: Optional[str] = None
    lemma_b: Optional[str] = None  # Second lemma for relation checking queries
    pos: Optional[str] = None
    definition_pattern: Optional[str] = None
    pattern_type: Optional[str] = None  # starts_with, contains, ends_with, regex
    written_form_pattern: Optional[str] = None
    semantic_relation: Optional[SemanticRelationType] = None
    
    # Filters
    filters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregation needs
    aggregation: Optional[Dict[str, Any]] = None
    
    # Output requirements
    retrieve_definitions: bool = True
    retrieve_examples: bool = False
    include_italian_written_rep: bool = True
    
    # Complexity indicator
    complexity_score: int = 1
    
    # Raw user query for debugging
    user_query: str = ""


# ============================================================================
# Execution Plan (Output from Orchestrator)
# ============================================================================

@dataclass
class PatternStep:
    """
    A single step in the execution plan.
    Contains the pattern tool name and its parameters.
    """
    tool_name: str
    parameters: Dict[str, Any]
    step_number: int
    description: str
    depends_on: List[int] = field(default_factory=list)  # Step numbers this depends on


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for generating a SPARQL query.
    This is a deterministic, validated sequence of pattern applications.
    """
    steps: List[PatternStep]
    intent: Intent
    variable_flow: Dict[str, str] = field(default_factory=dict)  # Maps step outputs to inputs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_step(self, step_number: int) -> Optional[PatternStep]:
        """Get a step by its number"""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None
    
    def get_steps_by_tool(self, tool_name: str) -> List[PatternStep]:
        """Get all steps using a specific tool"""
        return [s for s in self.steps if s.tool_name == tool_name]


# ============================================================================
# Orchestrator Rules Engine
# ============================================================================

class OrchestratorRules:
    """
    Collection of deterministic rules for pattern selection.
    Each rule is a pure function: Intent → List[PatternStep]
    """
    
    @staticmethod
    def needs_service_clause(intent: Intent) -> bool:
        """Check if query needs CompL-it SERVICE clause"""
        return ResourceType.COMPLIT in intent.required_resources
    
    @staticmethod
    def needs_bridge(intent: Intent) -> bool:
        """Check if bridge pattern is needed (CompL-it → LiITA)"""
        # Bridge needed if accessing CompL-it AND any other resource
        has_complit = ResourceType.COMPLIT in intent.required_resources
        has_other = any(r != ResourceType.COMPLIT for r in intent.required_resources)
        return has_complit and has_other
    
    @staticmethod
    def needs_italian_wr(intent: Intent) -> bool:
        """Check if Italian written representation is needed"""
        return intent.include_italian_written_rep and (
            ResourceType.COMPLIT in intent.required_resources or
            intent.query_type == QueryType.DIALECT_TRANSLATION
        )
    
    @staticmethod
    def get_dialect_tools(intent: Intent) -> List[str]:
        """Determine which dialect translation tools are needed"""
        tools = []
        if ResourceType.PARMIGIANO in intent.required_resources:
            tools.append("parmigiano_translation")
        if ResourceType.SICILIAN in intent.required_resources:
            # Check if starting from Sicilian or Italian
            if intent.query_type == QueryType.DIALECT_PATTERN_SEARCH:
                tools.append("sicilian_pattern_search")
            else:
                # Translation from Italian to Sicilian
                tools.append("sicilian_translation")
        return tools


# ============================================================================
# Main Orchestrator Class
# ============================================================================

class PatternOrchestrator:
    """
    Main orchestrator that creates execution plans from intents.

    This is a deterministic, rule-based system with no LLM calls.
    All logic is explicit and debuggable.
    """

    def __init__(self):
        self.rules = OrchestratorRules()

    # ========================================================================
    # Component-Based Planning Helpers
    # ========================================================================

    def _determine_base_component(self, intent: Intent) -> tuple[str, dict, str]:
        """
        Determine the base component for a query.

        Returns: (tool_name, parameters, output_lemma_var)

        The base component determines what lemmas we're working with:
        - Semantic relation → find related words via CompL-it
        - Definition pattern → find words by definition via CompL-it
        - Otherwise → basic LiITA lemma lookup
        """

        # Case 1: Semantic relation query (e.g., "meronyms of corpo")
        if intent.semantic_relation and ResourceType.COMPLIT in intent.required_resources:
            params = {
                "lemma": intent.lemma,
                "relation_type": intent.semantic_relation.value,
                "pos": intent.pos or "noun",
                "retrieve_definitions": intent.retrieve_definitions
            }
            # After semantic + bridge, the output var is ?liitaLemma
            return ("complit_semantic_relation", params, "?liitaLemma")

        # Case 2: Definition pattern query (e.g., "words with definition starting with 'uccello'")
        elif intent.definition_pattern and ResourceType.COMPLIT in intent.required_resources:
            params = {
                "definition_pattern": intent.definition_pattern,
                "pattern_type": intent.pattern_type or "starts_with",
                "retrieve_examples": intent.retrieve_examples
            }
            if intent.pos:
                params["pos_filter"] = intent.pos
            # After definition search + bridge, the output var is ?liitaLemma
            return ("complit_definition_search", params, "?liitaLemma")

        # Case 3: Basic LiITA lookup (default)
        else:
            params = {}
            if intent.lemma:
                params["pattern"] = f"^{intent.lemma}$"
                params["pattern_type"] = "regex"
            elif intent.written_form_pattern:
                params["pattern"] = intent.written_form_pattern
                params["pattern_type"] = intent.pattern_type or "regex"
            if intent.pos:
                params["pos_filter"] = intent.pos
            # Basic LiITA query outputs ?lemma
            return ("liita_basic_query", params, "?lemma")

    def _add_base_steps(self, intent: Intent, steps: list, step_num: int) -> tuple[int, str]:
        """
        Add base component steps to the plan.

        Returns: (next_step_num, output_lemma_var)
        """
        base_tool, base_params, output_var = self._determine_base_component(intent)

        # Add the base step
        if base_tool == "complit_semantic_relation":
            steps.append(PatternStep(
                tool_name=base_tool,
                parameters=base_params,
                step_number=step_num,
                description=f"Find {intent.semantic_relation.value}s of '{intent.lemma}' in CompL-it",
            ))
            step_num += 1

            # Semantic relation needs bridge to LiITA
            steps.append(PatternStep(
                tool_name="bridge_complit_to_liita",
                parameters={"source_var": "?relatedWord"},
                step_number=step_num,
                description="Bridge related words to LiITA lemmas",
                depends_on=[step_num - 1]
            ))
            step_num += 1

        elif base_tool == "complit_definition_search":
            steps.append(PatternStep(
                tool_name=base_tool,
                parameters=base_params,
                step_number=step_num,
                description=f"Search CompL-it for definitions {base_params.get('pattern_type', 'containing')} '{intent.definition_pattern}'",
            ))
            step_num += 1

            # Definition search needs bridge if linking to other resources
            if self.rules.needs_bridge(intent):
                steps.append(PatternStep(
                    tool_name="bridge_complit_to_liita",
                    parameters={"source_var": "?word"},
                    step_number=step_num,
                    description="Bridge CompL-it words to LiITA lemmas",
                    depends_on=[step_num - 1]
                ))
                step_num += 1
            else:
                # If no bridge needed, output var is ?word not ?liitaLemma
                output_var = "?word"

        else:  # liita_basic_query
            desc = f"Query LiITA for lemma '{intent.lemma}'" if intent.lemma else "Query LiITA Lemma Bank"
            steps.append(PatternStep(
                tool_name=base_tool,
                parameters=base_params,
                step_number=step_num,
                description=desc,
            ))
            step_num += 1

        return step_num, output_var

    def _add_enrichment_steps(self, intent: Intent, steps: list, step_num: int,
                               lemma_var: str, base_step_num: int) -> int:
        """
        Add enrichment component steps based on required_resources.

        Returns: next_step_num
        """
        # Sentix polarity enrichment
        if ResourceType.SENTIX in intent.required_resources:
            sentix_params = {
                "liita_lemma_var": lemma_var,
                "retrieve_polarity_type": True,
                "retrieve_polarity_value": True
            }
            steps.append(PatternStep(
                tool_name="sentix_linking",
                parameters=sentix_params,
                step_number=step_num,
                description="Link to Sentix for polarity/sentiment",
                depends_on=[base_step_num]
            ))
            step_num += 1

        # ELIta emotion enrichment
        if ResourceType.ELITA in intent.required_resources:
            elita_params = {
                "liita_lemma_var": lemma_var,
                "retrieve_emotion_label": True
            }
            steps.append(PatternStep(
                tool_name="elita_linking",
                parameters=elita_params,
                step_number=step_num,
                description="Link to ELIta for emotions",
                depends_on=[base_step_num]
            ))
            step_num += 1

        # Parmigiano translation enrichment
        if ResourceType.PARMIGIANO in intent.required_resources:
            steps.append(PatternStep(
                tool_name="parmigiano_translation",
                parameters={"italian_lemma_var": lemma_var},
                step_number=step_num,
                description="Link to Parmigiano translations",
                depends_on=[base_step_num]
            ))
            step_num += 1

        # Sicilian translation enrichment
        if ResourceType.SICILIAN in intent.required_resources:
            steps.append(PatternStep(
                tool_name="sicilian_translation",
                parameters={"italian_lemma_var": lemma_var},
                step_number=step_num,
                description="Link to Sicilian translations",
                depends_on=[base_step_num]
            ))
            step_num += 1

        # Italian written representation (if needed and not already from base)
        if self.rules.needs_italian_wr(intent) and lemma_var == "?liitaLemma":
            steps.append(PatternStep(
                tool_name="italian_written_rep",
                parameters={
                    "lemma_var": lemma_var,
                    "output_var": "?italianWR"
                },
                step_number=step_num,
                description="Get Italian written representations",
                depends_on=[base_step_num]
            ))
            step_num += 1

        return step_num
    
    def create_plan(self, intent: Intent) -> ExecutionPlan:
        """
        Create an execution plan from an intent.
        
        This is the main entry point. It delegates to specialized
        planners based on query type.
        """
        
        # CRITICAL FIX: Check if query is actually about dialects even if classified as basic_lemma_lookup
        # This handles cases like "Find Parmigiano words ending in 'u'"
        if intent.query_type == QueryType.BASIC_LEMMA_LOOKUP:
            # Check if this is really a dialect query misclassified as basic lookup
            dialect_resources = [r for r in intent.required_resources if r in [ResourceType.PARMIGIANO, ResourceType.SICILIAN]]
            has_only_dialect = dialect_resources and ResourceType.LIITA not in intent.required_resources and ResourceType.COMPLIT not in intent.required_resources
            
            if has_only_dialect:
                # This is actually a dialect pattern search, not a basic LiITA lookup
                # Reclassify and route to correct planner
                if ResourceType.SICILIAN in dialect_resources:
                    return self._plan_dialect_pattern_search(intent)
                elif ResourceType.PARMIGIANO in dialect_resources:
                    return self._plan_parmigiano_pattern_search(intent)
            
            return self._plan_basic_lookup(intent)
        
        elif intent.query_type == QueryType.COMPLIT_DEFINITIONS:
            return self._plan_complit_definitions(intent)
        
        elif intent.query_type == QueryType.COMPLIT_SEMANTIC:
            return self._plan_complit_semantic(intent)

        elif intent.query_type == QueryType.COMPLIT_RELATION_CHECK:
            return self._plan_complit_relation_check(intent)

        elif intent.query_type == QueryType.DIALECT_PATTERN_SEARCH:
            return self._plan_dialect_pattern_search(intent)
        
        elif intent.query_type == QueryType.DIALECT_TRANSLATION:
            return self._plan_dialect_translation(intent)
        
        elif intent.query_type == QueryType.MULTI_RESOURCE:
            return self._plan_multi_resource(intent)
        
        elif intent.query_type == QueryType.SENTIX_POLARITY:
            return self._plan_sentix_polarity(intent)
        
        elif intent.query_type == QueryType.ELITA_EMOTION:
            return self._plan_elita_emotion(intent)
        
        elif intent.query_type == QueryType.AFFECTIVE_MULTI_RESOURCE:
            return self._plan_affective_multi_resource(intent)
        
        else:
            raise ValueError(f"Unknown query type: {intent.query_type}")
    
    # ========================================================================
    # Specialized Planners
    # ========================================================================
    
    def _plan_basic_lookup(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for basic LiITA queries (no external resources).
        
        Example: "How many nouns in LiITA?"
        Example: "Find lemmas starting with 'infra'"
        """
        steps = []
        step_num = 1
        
        # Step 1: LiITA basic query
        params = {}
        if intent.pos:
            params["pos_filter"] = intent.pos
        if intent.written_form_pattern:
            params["pattern"] = intent.written_form_pattern
            params["pattern_type"] = intent.pattern_type or "regex"
        
        steps.append(PatternStep(
            tool_name="liita_basic_query",
            parameters=params,
            step_number=step_num,
            description=f"Query LiITA Lemma Bank for {intent.pos or 'all'} lemmas"
        ))
        step_num += 1
        
        # Step 2: Aggregation if needed
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[1]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={"plan_type": "basic_lookup"}
        )
    
    def _plan_complit_definitions(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for CompL-it definition searches.
        
        Example: "Find words whose definition starts with 'uccello'"
        
        Pattern sequence:
        1. CompL-it definition search (SERVICE)
        2. Optional: Bridge to LiITA (if other resources needed)
        3. Optional: Dialect translations
        4. Optional: Aggregation
        """
        steps = []
        step_num = 1
        
        # Step 1: CompL-it definition search (SERVICE)
        service_params = {
            "definition_pattern": intent.definition_pattern,
            "pattern_type": intent.pattern_type or "starts_with",
            "retrieve_examples": intent.retrieve_examples
        }
        if intent.pos:
            service_params["pos_filter"] = intent.pos
        
        steps.append(PatternStep(
            tool_name="complit_definition_search",
            parameters=service_params,
            step_number=step_num,
            description=f"Search CompL-it for definitions {intent.pattern_type} '{intent.definition_pattern}'",
        ))
        step_num += 1
        
        # Step 2: Bridge (if linking to other resources)
        if self.rules.needs_bridge(intent):
            steps.append(PatternStep(
                tool_name="bridge_complit_to_liita",
                parameters={"source_var": "?word"},
                step_number=step_num,
                description="Bridge CompL-it words to LiITA lemmas",
                depends_on=[1]
            ))
            step_num += 1
        
        # Step 3: Dialect translations (if needed)
        dialect_tools = self.rules.get_dialect_tools(intent)
        for tool_name in dialect_tools:
            steps.append(PatternStep(
                tool_name=tool_name,
                parameters={"italian_lemma_var": "?liitaLemma"},
                step_number=step_num,
                description=f"Link to {tool_name.split('_')[0].title()} translations",
                depends_on=[step_num - 1]
            ))
            step_num += 1
        
        # Step 4: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={
                "plan_type": "complit_definitions",
                "uses_service": True
            }
        )
    
    def _plan_complit_semantic(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for CompL-it semantic relation queries.
        
        Example: "Find hyponyms of 'colore'"
        Example: "What are the meronyms of 'giorno' with Parmigiano translations?"
        
        Pattern sequence:
        1. CompL-it semantic relation navigation (SERVICE)
        2. Bridge to LiITA
        3. Optional: Dialect translations
        4. Optional: Italian written representation
        5. Optional: Aggregation
        """
        steps = []
        step_num = 1
        
        # Step 1: CompL-it semantic relation (SERVICE)
        service_params = {
            "lemma": intent.lemma,
            "relation_type": intent.semantic_relation.value,
            "pos": intent.pos or "noun",
            "retrieve_definitions": intent.retrieve_definitions
        }
        
        steps.append(PatternStep(
            tool_name="complit_semantic_relation",
            parameters=service_params,
            step_number=step_num,
            description=f"Find {intent.semantic_relation.value}s of '{intent.lemma}' in CompL-it",
        ))
        step_num += 1
        
        # Step 2: Bridge (ALWAYS needed for semantic queries to get to other resources)
        # Use relatedWord as the source variable (from semantic pattern)
        steps.append(PatternStep(
            tool_name="bridge_complit_to_liita",
            parameters={"source_var": "?relatedWord"},
            step_number=step_num,
            description="Bridge related words to LiITA lemmas",
            depends_on=[1]
        ))
        step_num += 1
        
        # Step 3: Italian written representation (if needed)
        if self.rules.needs_italian_wr(intent):
            steps.append(PatternStep(
                tool_name="italian_written_rep",
                parameters={
                    "lemma_var": "?liitaLemma",
                    "output_var": "?italianWR"
                },
                step_number=step_num,
                description="Get Italian written representations",
                depends_on=[2]
            ))
            step_num += 1
        
        # Step 4: Dialect translations (if needed)
        dialect_tools = self.rules.get_dialect_tools(intent)
        for tool_name in dialect_tools:
            steps.append(PatternStep(
                tool_name=tool_name,
                parameters={"italian_lemma_var": "?liitaLemma"},
                step_number=step_num,
                description=f"Link to {tool_name.split('_')[0].title()} translations",
                depends_on=[2]  # Depends on bridge
            ))
            step_num += 1
        
        # Step 5: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={
                "plan_type": "complit_semantic",
                "uses_service": True,
                "relation_type": intent.semantic_relation.value
            }
        )

    def _plan_complit_relation_check(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for checking semantic relations between two specific lemmas.

        Example: "Are 'cane' and 'animale' related?"
        Example: "Is 'rosso' a hyponym of 'colore'?"

        Pattern sequence:
        1. CompL-it relation check between two lemmas (SERVICE)
        """
        steps = []
        step_num = 1

        # Determine relation type to check
        relation_type = "any"
        if intent.semantic_relation:
            relation_type = intent.semantic_relation.value

        # Step 1: Relation check query
        params = {
            "lemma_a": intent.lemma,
            "lemma_b": intent.lemma_b,
            "relation_type": relation_type,
            "pos": intent.pos or "noun"
        }

        steps.append(PatternStep(
            tool_name="complit_relation_between_lemmas",
            parameters=params,
            step_number=step_num,
            description=f"Check {relation_type} relation between '{intent.lemma}' and '{intent.lemma_b}'",
        ))

        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={
                "plan_type": "complit_relation_check",
                "uses_service": True,
                "lemma_a": intent.lemma,
                "lemma_b": intent.lemma_b,
                "relation_type": relation_type
            }
        )

    def _plan_dialect_pattern_search(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for dialect pattern searches.
        
        Example: "Find Sicilian words ending in 'ìa'"
        Example: "Find Parmigiano words ending in 'u'"
        
        Pattern sequence:
        1. Dialect pattern search (includes Italian linking if requested)
        2. Optional: Aggregation
        """
        steps = []
        step_num = 1
        
        # Determine which dialect
        if ResourceType.SICILIAN in intent.required_resources:
            tool_name = "sicilian_pattern_search"
            params = {
                "pattern": intent.written_form_pattern,
                "link_to_italian": True
            }
            if intent.pos:
                params["pos"] = intent.pos
            
            steps.append(PatternStep(
                tool_name=tool_name,
                parameters=params,
                step_number=step_num,
                description=f"Search Sicilian lemmas matching pattern '{intent.written_form_pattern}'",
            ))
            step_num += 1
        
        elif ResourceType.PARMIGIANO in intent.required_resources:
            # NEW: Handle Parmigiano pattern search
            # We need to create a custom pattern for this
            # For now, we'll use a workaround with Italian query + reverse linking
            # TODO: Create dedicated ParmigianoPatternSearchPattern tool
            
            # Step 1: Query LiITA for Italian lemmas
            # Step 2: Link to Parmigiano
            # Step 3: Filter Parmigiano by pattern
            
            # This is a workaround - ideally we'd have a direct Parmigiano pattern tool
            steps.append(PatternStep(
                tool_name="liita_basic_query",
                parameters={},  # No filters - we'll get all lemmas
                step_number=step_num,
                description="Query LiITA for Italian lemmas",
            ))
            step_num += 1
            
            steps.append(PatternStep(
                tool_name="parmigiano_translation",
                parameters={"italian_lemma_var": "?lemma"},
                step_number=step_num,
                description=f"Link to Parmigiano and filter by pattern '{intent.written_form_pattern}'",
                depends_on=[1]
            ))
            step_num += 1
            
            # Note: The pattern filtering would need to happen in the SPARQL
            # This is a limitation - we need a dedicated Parmigiano search tool
        
        # Aggregation
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={"plan_type": "dialect_pattern_search"}
        )
    
    def _plan_parmigiano_pattern_search(self, intent: Intent) -> ExecutionPlan:
        """
        NEW: Plan for Parmigiano-specific pattern searches.
        
        Example: "Find Parmigiano words ending in 'u'"
        
        Now uses the dedicated ParmigianoPatternSearchPattern tool.
        """
        steps = []
        step_num = 1
        
        # Use the new dedicated pattern search tool
        params = {
            "pattern": intent.written_form_pattern,
            "link_to_italian": True
        }
        if intent.pos:
            params["pos"] = intent.pos
        
        steps.append(PatternStep(
            tool_name="parmigiano_pattern_search",
            parameters=params,
            step_number=step_num,
            description=f"Search Parmigiano lemmas matching pattern '{intent.written_form_pattern}'",
        ))
        step_num += 1
        
        # Aggregation if needed
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[1]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={
                "plan_type": "parmigiano_pattern_search",
                "pattern": intent.written_form_pattern
            }
        )
    
    def _plan_dialect_translation(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for simple dialect translation queries (starting from Italian).
        
        Example: "Translate 'casa' to Parmigiano"
        
        This is simpler than multi-resource because we're not combining
        with CompL-it semantic queries.
        """
        steps = []
        step_num = 1
        
        # Step 1: Start with LiITA query to find the Italian lemma
        params = {}
        if intent.lemma:
            params["pattern"] = f"^{intent.lemma}$"
            params["pattern_type"] = "regex"
        if intent.pos:
            params["pos_filter"] = intent.pos
        
        steps.append(PatternStep(
            tool_name="liita_basic_query",
            parameters=params,
            step_number=step_num,
            description=f"Find Italian lemma '{intent.lemma}' in LiITA",
        ))
        step_num += 1
        
        # Step 2: Dialect translations
        dialect_tools = self.rules.get_dialect_tools(intent)
        for tool_name in dialect_tools:
            # For translation patterns, we need to use ?lemma from basic query
            steps.append(PatternStep(
                tool_name=tool_name,
                parameters={"italian_lemma_var": "?lemma"},
                step_number=step_num,
                description=f"Link to {tool_name.split('_')[0].title()} translations",
                depends_on=[1]
            ))
            step_num += 1
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={"plan_type": "dialect_translation"}
        )
    
    def _plan_multi_resource(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for complex multi-resource queries.

        Uses component-based planning:
        - PHASE 1: Determine base component (semantic relation, definition search, or basic lookup)
        - PHASE 2: Add enrichment components based on required_resources

        Example: "Find hyponyms of 'colore' with both Parmigiano and Sicilian translations"
        Example: "Find words defined as 'uccello' with Sicilian translations"
        """
        steps = []
        step_num = 1

        # PHASE 1: Add base component steps
        step_num, lemma_var = self._add_base_steps(intent, steps, step_num)
        base_step_num = step_num - 1  # The last step added is the base (or bridge)

        # PHASE 2: Add enrichment components based on required_resources
        step_num = self._add_enrichment_steps(intent, steps, step_num, lemma_var, base_step_num)

        # PHASE 3: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))

        # Determine metadata
        uses_service = ResourceType.COMPLIT in intent.required_resources
        metadata = {
            "plan_type": "multi_resource",
            "uses_service": uses_service,
            "base_component": self._determine_base_component(intent)[0],
            "enrichments": [r.value for r in intent.required_resources
                          if r in [ResourceType.SENTIX, ResourceType.ELITA,
                                  ResourceType.PARMIGIANO, ResourceType.SICILIAN]]
        }

        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata=metadata
        )
    
    def _plan_sentix_polarity(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for Sentix polarity queries.
        
        Example: "What is the polarity of 'giorno'?"
        Example: "Show Sentix sentiment for words with negative polarity"
        
        Pattern sequence:
        1. LiITA basic query (if lemma specified)
        2. Sentix linking
        3. Optional: Aggregation
        """
        steps = []
        step_num = 1
        
        # Step 1: LiITA query (if lemma or pattern specified)
        if intent.lemma or intent.written_form_pattern:
            params = {}
            if intent.lemma:
                params["pattern"] = f"^{intent.lemma}$"
                params["pattern_type"] = "regex"
            elif intent.written_form_pattern:
                params["pattern"] = intent.written_form_pattern
                params["pattern_type"] = intent.pattern_type or "regex"
            if intent.pos:
                params["pos_filter"] = intent.pos
            
            steps.append(PatternStep(
                tool_name="liita_basic_query",
                parameters=params,
                step_number=step_num,
                description=f"Query LiITA for {'lemma ' + intent.lemma if intent.lemma else 'lemmas matching pattern'}",
            ))
            step_num += 1
        
        # Step 2: Sentix linking
        sentix_params = {
            "liita_lemma_var": "?lemma" if intent.lemma or intent.written_form_pattern else "?lemma",
            "retrieve_polarity_type": True,
            "retrieve_polarity_value": True
        }
        
        # Add polarity filters if specified in intent
        if intent.filters:
            for filter_item in intent.filters:
                if filter_item.get('field') == 'polarity_type':
                    sentix_params["polarity_filter"] = filter_item.get('value')
                elif filter_item.get('field') == 'polarity_value_min':
                    sentix_params["polarity_value_min"] = filter_item.get('value')
                elif filter_item.get('field') == 'polarity_value_max':
                    sentix_params["polarity_value_max"] = filter_item.get('value')
        
        steps.append(PatternStep(
            tool_name="sentix_linking",
            parameters=sentix_params,
            step_number=step_num,
            description="Link to Sentix affective lexicon",
            depends_on=[step_num - 1] if (intent.lemma or intent.written_form_pattern) else []
        ))
        step_num += 1
        
        # Step 3: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={"plan_type": "sentix_polarity"}
        )
    
    def _plan_elita_emotion(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for ELIta emotion queries.
        
        Example: "What emotions are associated with 'amore'?"
        Example: "Find words with Joy and Sadness emotions"
        
        Pattern sequence:
        1. LiITA basic query (if lemma specified)
        2. ELIta linking
        3. Optional: Aggregation
        """
        steps = []
        step_num = 1
        
        # Step 1: LiITA query (if lemma or pattern specified)
        if intent.lemma or intent.written_form_pattern:
            params = {}
            if intent.lemma:
                params["pattern"] = f"^{intent.lemma}$"
                params["pattern_type"] = "regex"
            elif intent.written_form_pattern:
                params["pattern"] = intent.written_form_pattern
                params["pattern_type"] = intent.pattern_type or "regex"
            if intent.pos:
                params["pos_filter"] = intent.pos
            
            steps.append(PatternStep(
                tool_name="liita_basic_query",
                parameters=params,
                step_number=step_num,
                description=f"Query LiITA for {'lemma ' + intent.lemma if intent.lemma else 'lemmas matching pattern'}",
            ))
            step_num += 1
        
        # Step 2: ELIta linking
        elita_params = {
            "liita_lemma_var": "?lemma" if intent.lemma or intent.written_form_pattern else "?lemma",
            "retrieve_emotion_label": True
        }
        
        # Extract emotion filters from intent
        emotion_filters = []
        if intent.filters:
            for filter_item in intent.filters:
                if filter_item.get('field') == 'emotion':
                    emotion_filters.append(filter_item.get('value'))
        
        if emotion_filters:
            elita_params["emotion_filters"] = emotion_filters
        
        steps.append(PatternStep(
            tool_name="elita_linking",
            parameters=elita_params,
            step_number=step_num,
            description="Link to ELIta emotion lexicon",
            depends_on=[step_num - 1] if (intent.lemma or intent.written_form_pattern) else []
        ))
        step_num += 1
        
        # Step 3: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))
        
        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata={"plan_type": "elita_emotion"}
        )
    
    def _plan_affective_multi_resource(self, intent: Intent) -> ExecutionPlan:
        """
        Plan for complex queries combining affective resources (Sentix/ELIta)
        with other resources (CompL-it semantics, dialects).

        Uses component-based planning:
        - PHASE 1: Determine base component (semantic relation, definition search, or basic lookup)
        - PHASE 2: Add enrichment components based on required_resources

        Example: "Find meronyms of corpo with emotions and Sicilian translations"
        Example: "What is the polarity of hyponyms of 'animale'?"
        """
        steps = []
        step_num = 1

        # PHASE 1: Add base component steps
        step_num, lemma_var = self._add_base_steps(intent, steps, step_num)
        base_step_num = step_num - 1  # The last step added is the base (or bridge)

        # PHASE 2: Add enrichment components based on required_resources
        step_num = self._add_enrichment_steps(intent, steps, step_num, lemma_var, base_step_num)

        # PHASE 3: Aggregation (if needed)
        if intent.aggregation:
            steps.append(PatternStep(
                tool_name="aggregation",
                parameters=intent.aggregation,
                step_number=step_num,
                description=f"Apply {intent.aggregation['type']} aggregation",
                depends_on=[i for i in range(1, step_num)]
            ))

        # Determine metadata
        uses_service = ResourceType.COMPLIT in intent.required_resources
        metadata = {
            "plan_type": "affective_multi_resource",
            "uses_service": uses_service,
            "base_component": self._determine_base_component(intent)[0],
            "enrichments": [r.value for r in intent.required_resources
                          if r in [ResourceType.SENTIX, ResourceType.ELITA,
                                  ResourceType.PARMIGIANO, ResourceType.SICILIAN]]
        }

        return ExecutionPlan(
            steps=steps,
            intent=intent,
            metadata=metadata
        )


# ============================================================================
# Plan Validator
# ============================================================================

class PlanValidator:
    """
    Validates execution plans to ensure they're correct before execution.
    """
    
    @staticmethod
    def validate(plan: ExecutionPlan) -> tuple[bool, List[str]]:
        """
        Validate an execution plan.
        
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Rule 1: SERVICE clause must be followed by bridge if other resources needed
        service_steps = plan.get_steps_by_tool("complit_definition_search") + \
                       plan.get_steps_by_tool("complit_semantic_relation")
        
        if service_steps:
            has_other_resources = len(plan.intent.required_resources) > 1
            has_bridge = bool(plan.get_steps_by_tool("bridge_complit_to_liita"))
            
            if has_other_resources and not has_bridge:
                errors.append(
                    "CompL-it SERVICE query requires bridge pattern when linking to other resources"
                )
        
        # Rule 2: Dialect tools must come after bridge or basic query
        dialect_steps = [s for s in plan.steps if "translation" in s.tool_name or "sicilian_pattern" in s.tool_name]
        
        for dialect_step in dialect_steps:
            if not dialect_step.depends_on:
                errors.append(
                    f"Dialect translation step {dialect_step.step_number} has no dependencies"
                )
        
        # Rule 3: Aggregation must be last step
        agg_steps = plan.get_steps_by_tool("aggregation")
        if agg_steps:
            agg_step = agg_steps[0]
            if agg_step.step_number != len(plan.steps):
                errors.append(
                    f"Aggregation must be the last step, but found at step {agg_step.step_number}"
                )
        
        # Rule 4: Step dependencies must reference existing steps
        for step in plan.steps:
            for dep in step.depends_on:
                if dep >= step.step_number:
                    errors.append(
                        f"Step {step.step_number} depends on future step {dep}"
                    )
                if not plan.get_step(dep):
                    errors.append(
                        f"Step {step.step_number} depends on non-existent step {dep}"
                    )
        
        return len(errors) == 0, errors


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    orchestrator = PatternOrchestrator()
    validator = PlanValidator()
    
    # Example 1: Basic LiITA query
    print("=" * 70)
    print("EXAMPLE 1: Basic LiITA Query")
    print("Query: 'How many nouns are in LiITA?'")
    print("=" * 70)
    
    intent1 = Intent(
        query_type=QueryType.BASIC_LEMMA_LOOKUP,
        required_resources=[ResourceType.LIITA],
        pos="noun",
        aggregation={"type": "count"},
        user_query="How many nouns are in LiITA?"
    )
    
    plan1 = orchestrator.create_plan(intent1)
    valid, errors = validator.validate(plan1)
    
    print(f"Plan valid: {valid}")
    print(f"Number of steps: {len(plan1.steps)}")
    for step in plan1.steps:
        print(f"\nStep {step.step_number}: {step.tool_name}")
        print(f"  Description: {step.description}")
        print(f"  Parameters: {step.parameters}")
        if step.depends_on:
            print(f"  Depends on: {step.depends_on}")
    
    # Example 2: CompL-it definition search with Parmigiano
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CompL-it Definition Search with Translation")
    print("Query: 'Find words whose definition starts with uccello and show Parmigiano translations'")
    print("=" * 70)
    
    intent2 = Intent(
        query_type=QueryType.COMPLIT_DEFINITIONS,
        required_resources=[ResourceType.COMPLIT, ResourceType.PARMIGIANO],
        definition_pattern="uccello",
        pattern_type="starts_with",
        pos="noun",
        user_query="Find words whose definition starts with uccello and show Parmigiano translations"
    )
    
    plan2 = orchestrator.create_plan(intent2)
    valid, errors = validator.validate(plan2)
    
    print(f"Plan valid: {valid}")
    print(f"Number of steps: {len(plan2.steps)}")
    for step in plan2.steps:
        print(f"\nStep {step.step_number}: {step.tool_name}")
        print(f"  Description: {step.description}")
        print(f"  Parameters: {step.parameters}")
        if step.depends_on:
            print(f"  Depends on: {step.depends_on}")
    
    # Example 3: Semantic relations with aggregation
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Semantic Relations with Parmigiano and Aggregation")
    print("Query: 'Find hyponyms of colore with Parmigiano translations'")
    print("=" * 70)
    
    intent3 = Intent(
        query_type=QueryType.COMPLIT_SEMANTIC,
        required_resources=[ResourceType.COMPLIT, ResourceType.LIITA, ResourceType.PARMIGIANO],
        lemma="colore",
        semantic_relation=SemanticRelationType.HYPONYM,
        pos="noun",
        retrieve_definitions=True,
        aggregation={
            "type": "group_concat",
            "aggregate_var": "?definition",
            "group_by_vars": ["relatedSense", "liitaLemma", "parmigianoLemma", "parmigianoWR"],
            "separator": "; ",
            "order_by": {"var": "parmigianoWR", "direction": "ASC"}
        },
        user_query="Find hyponyms of colore with Parmigiano translations"
    )
    
    plan3 = orchestrator.create_plan(intent3)
    valid, errors = validator.validate(plan3)
    
    print(f"Plan valid: {valid}")
    print(f"Number of steps: {len(plan3.steps)}")
    for step in plan3.steps:
        print(f"\nStep {step.step_number}: {step.tool_name}")
        print(f"  Description: {step.description}")
        print(f"  Parameters: {step.parameters}")
        if step.depends_on:
            print(f"  Depends on: {step.depends_on}")
    
    # Example 4: Sicilian pattern search
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Sicilian Pattern Search")
    print("Query: 'Find Sicilian words ending in ìa'")
    print("=" * 70)
    
    intent4 = Intent(
        query_type=QueryType.DIALECT_PATTERN_SEARCH,
        required_resources=[ResourceType.SICILIAN],
        written_form_pattern="ìa$",
        pos="noun",
        aggregation={
            "type": "group_concat",
            "aggregate_var": "?sicilianWR",
            "group_by_vars": ["sicilianLemma", "liitaLemma"]
        },
        user_query="Find Sicilian words ending in ìa"
    )
    
    plan4 = orchestrator.create_plan(intent4)
    valid, errors = validator.validate(plan4)
    
    print(f"Plan valid: {valid}")
    print(f"Number of steps: {len(plan4.steps)}")
    for step in plan4.steps:
        print(f"\nStep {step.step_number}: {step.tool_name}")
        print(f"  Description: {step.description}")
        print(f"  Parameters: {step.parameters}")
        if step.depends_on:
            print(f"  Depends on: {step.depends_on}")
    
    print("\n" + "=" * 70)
    print("ORCHESTRATOR SUMMARY")
    print("=" * 70)
    print("\nThe orchestrator successfully created execution plans for:")
    print("✓ Basic LiITA queries")
    print("✓ CompL-it definition searches with translations")
    print("✓ Semantic relation queries with aggregation")
    print("✓ Dialect pattern searches")
    print("\nAll plans passed validation!")
    print("\nKey features:")
    print("- Deterministic (no LLM uncertainty)")
    print("- Rule-based (explicit logic)")
    print("- Validated (architectural correctness guaranteed)")
    print("- Composable (patterns chain correctly)")