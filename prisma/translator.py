"""
PRISMA Translator
=================

Complete integration of all components into a working NL2SPARQL system.

Components integrated:
1. Intent Analyzer (LLM-based classification)
2. Pattern Orchestrator (Rule-based planning)
3. Pattern Tools (Template-based generation)
4. Pattern Assembler (SPARQL composition)

Part of PRISMA per LiITA - Pattern-based Rules for Intent-driven SPARQL
with Multiple-resource Assembly for the LiITA knowledge base.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import json
import time

from .pattern_tools import PatternToolRegistry, PatternFragment
from .orchestrator import (
    PatternOrchestrator, Intent, QueryType, ResourceType,
    SemanticRelationType, ExecutionPlan
)
from .assembler import PatternAssembler, AssemblyResult
from .intent_analyzer import IntentAnalyzer


# ============================================================================
# Intent Converter (Dict → Intent Object)
# ============================================================================

class IntentConverter:
    """
    Converts intent dictionaries (from LLM) into Intent objects (for orchestrator).

    This handles the translation between the LLM's JSON output and the
    type-safe Intent dataclass.
    """

    @staticmethod
    def dict_to_intent(intent_dict: Dict) -> Intent:
        """
        Convert intent dictionary from LLM into Intent object.

        Handles type conversions, enum parsing, and default values.
        """
        # Parse query type
        query_type_str = intent_dict.get('query_type', 'basic_lemma_lookup')
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.BASIC_LEMMA_LOOKUP

        # Parse resources
        resources = []
        # Normalize resource names (handle common variations from LLM)
        resource_normalization = {
            'siciliano': 'sicilian',
            'parmigiano': 'parmigiano',
            'liita': 'liita_lemma_bank',
            'liita_lemma_bank': 'liita_lemma_bank',
            'complit': 'complit',
            'sentix': 'sentix',
            'elita': 'elita',
        }

        for r in intent_dict.get('required_resources', []):
            normalized_r = resource_normalization.get(r.lower(), r.lower())
            try:
                resources.append(ResourceType(normalized_r))
            except ValueError:
                try:
                    resources.append(ResourceType(r))
                except ValueError:
                    continue

        if not resources:
            resources = [ResourceType.LIITA]

        # Parse semantic relation if present
        semantic_relation = None
        if intent_dict.get('semantic_relation'):
            try:
                semantic_relation = SemanticRelationType(intent_dict['semantic_relation'])
            except ValueError:
                pass

        # Create Intent object
        return Intent(
            query_type=query_type,
            required_resources=resources,
            lemma=intent_dict.get('lemma'),
            pos=intent_dict.get('pos'),
            definition_pattern=intent_dict.get('definition_pattern'),
            pattern_type=intent_dict.get('pattern_type'),
            written_form_pattern=intent_dict.get('written_form_pattern'),
            semantic_relation=semantic_relation,
            filters=intent_dict.get('filters', []),
            aggregation=intent_dict.get('aggregation'),
            retrieve_definitions=intent_dict.get('retrieve_definitions', True),
            retrieve_examples=intent_dict.get('retrieve_examples', False),
            include_italian_written_rep=intent_dict.get('include_italian_written_rep', True),
            complexity_score=intent_dict.get('complexity_score', 1),
            user_query=intent_dict.get('user_query', '')
        )


# ============================================================================
# Complete Translation Result
# ============================================================================

@dataclass
class TranslationResult:
    """
    Complete result from NL2SPARQL translation.

    Contains everything needed to understand, debug, and execute the query.
    """
    # User input
    user_query: str

    # LLM analysis
    intent_dict: Dict
    intent_warnings: List[str]

    # Planning
    execution_plan: ExecutionPlan
    plan_valid: bool
    plan_errors: List[str]

    # Assembly
    sparql_query: str
    assembly_warnings: List[str]
    variable_mappings: Dict[str, str]

    # Metadata
    total_steps: int
    uses_service: bool
    has_aggregation: bool
    complexity_score: int
    processing_time_ms: float

    # Success flag
    success: bool
    error_message: Optional[str] = None

    def __str__(self):
        """Pretty print the result"""
        return self.sparql_query

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'user_query': self.user_query,
            'sparql_query': self.sparql_query,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': {
                'total_steps': self.total_steps,
                'uses_service': self.uses_service,
                'has_aggregation': self.has_aggregation,
                'complexity_score': self.complexity_score,
                'processing_time_ms': self.processing_time_ms
            },
            'warnings': {
                'intent': self.intent_warnings,
                'assembly': self.assembly_warnings
            },
            'debug': {
                'intent': self.intent_dict,
                'plan_errors': self.plan_errors,
                'variable_mappings': self.variable_mappings
            }
        }


# ============================================================================
# Main Translator
# ============================================================================

class Translator:
    """
    Complete Natural Language to SPARQL translation system for LiITA.

    This is the main entry point that orchestrates all components:
    1. Intent Analysis (LLM)
    2. Pattern Orchestration (Rules)
    3. Pattern Assembly (Composition)

    Usage:
        from prisma import Translator, create_llm_client

        client = create_llm_client(provider="mistral", api_key="...")
        translator = Translator(client)
        result = translator.translate("Find hyponyms of colore")
        print(result.sparql_query)
    """

    def __init__(self, llm_client, verbose: bool = False):
        """
        Initialize the translator.

        Args:
            llm_client: Client implementing complete(prompt, system, temperature)
            verbose: If True, print detailed progress information
        """
        self.verbose = verbose

        # Initialize components
        self.analyzer = IntentAnalyzer(llm_client)
        self.orchestrator = PatternOrchestrator()
        self.registry = PatternToolRegistry()
        self.assembler = PatternAssembler(self.registry)
        self.converter = IntentConverter()

        self._log("PRISMA translator initialized")
        self._log(f"  - Intent Analyzer: ready")
        self._log(f"  - Pattern Orchestrator: ready")
        self._log(f"  - Pattern Tools: {len(self.registry.list_tools())} tools registered")
        self._log(f"  - Pattern Assembler: ready")

    def translate(
        self,
        user_query: str,
        include_examples: int = 5,
        validate_plan: bool = True
    ) -> TranslationResult:
        """
        Translate a natural language query into SPARQL.

        This is the main entry point for the system.

        Args:
            user_query: Natural language query from user
            include_examples: Number of few-shot examples for LLM (0-11)
            validate_plan: Whether to validate the execution plan

        Returns:
            TranslationResult with SPARQL query and metadata
        """
        start_time = time.time()

        try:
            self._log(f"\n{'='*70}")
            self._log(f"TRANSLATING: {user_query}")
            self._log(f"{'='*70}")

            # Step 1: Intent Analysis (LLM)
            self._log("\n[1/3] Analyzing intent with LLM...")
            intent_dict, intent_warnings = self.analyzer.analyze(
                user_query,
                include_examples=include_examples
            )

            self._log(f"  [OK] Query type: {intent_dict.get('query_type')}")
            self._log(f"  [OK] Resources: {intent_dict.get('required_resources')}")
            if intent_warnings:
                self._log(f"  [WARN] Warnings: {len(intent_warnings)}")

            # Convert to Intent object
            intent = self.converter.dict_to_intent(intent_dict)

            # Step 2: Pattern Orchestration (Rules)
            self._log("\n[2/3] Creating execution plan...")
            execution_plan = self.orchestrator.create_plan(intent)

            self._log(f"  [OK] Plan created with {len(execution_plan.steps)} steps:")
            for i, step in enumerate(execution_plan.steps, 1):
                self._log(f"    {i}. {step.tool_name}")

            # Validate plan if requested
            plan_valid = True
            plan_errors = []
            if validate_plan:
                from .orchestrator import PlanValidator
                validator = PlanValidator()
                plan_valid, plan_errors = validator.validate(execution_plan)
                if not plan_valid:
                    self._log(f"  [WARN] Plan validation failed: {len(plan_errors)} errors")
                else:
                    self._log(f"  [OK] Plan validated successfully")

            # Step 3: SPARQL Assembly (Composition)
            self._log("\n[3/3] Assembling SPARQL query...")
            assembly_result = self.assembler.assemble(execution_plan)

            self._log(f"  [OK] Query assembled")
            self._log(f"  [OK] Fragments: {len(assembly_result.fragments)}")
            self._log(f"  [OK] Uses SERVICE: {assembly_result.metadata['uses_service']}")
            self._log(f"  [OK] Has aggregation: {assembly_result.metadata['has_aggregation']}")

            if assembly_result.warnings:
                self._log(f"  [WARN] Assembly warnings: {len(assembly_result.warnings)}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Build result
            result = TranslationResult(
                user_query=user_query,
                intent_dict=intent_dict,
                intent_warnings=intent_warnings,
                execution_plan=execution_plan,
                plan_valid=plan_valid,
                plan_errors=plan_errors,
                sparql_query=assembly_result.sparql_query,
                assembly_warnings=assembly_result.warnings,
                variable_mappings=assembly_result.variable_mappings,
                total_steps=len(execution_plan.steps),
                uses_service=assembly_result.metadata['uses_service'],
                has_aggregation=assembly_result.metadata['has_aggregation'],
                complexity_score=intent_dict.get('complexity_score', 1),
                processing_time_ms=processing_time,
                success=True,
                error_message=None
            )

            self._log(f"\n[OK] Translation successful in {processing_time:.1f}ms")

            return result

        except Exception as e:
            # Handle errors gracefully
            processing_time = (time.time() - start_time) * 1000

            self._log(f"\n[FAIL] Translation failed: {str(e)}")

            return TranslationResult(
                user_query=user_query,
                intent_dict={},
                intent_warnings=[],
                execution_plan=None,
                plan_valid=False,
                plan_errors=[str(e)],
                sparql_query="",
                assembly_warnings=[],
                variable_mappings={},
                total_steps=0,
                uses_service=False,
                has_aggregation=False,
                complexity_score=0,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )

    def translate_batch(
        self,
        queries: List[str],
        include_examples: int = 5
    ) -> List[TranslationResult]:
        """
        Translate multiple queries in batch.

        Args:
            queries: List of natural language queries
            include_examples: Number of few-shot examples for LLM

        Returns:
            List of TranslationResult objects
        """
        results = []

        self._log(f"\n{'='*70}")
        self._log(f"BATCH TRANSLATION: {len(queries)} queries")
        self._log(f"{'='*70}")

        for i, query in enumerate(queries, 1):
            self._log(f"\nProcessing {i}/{len(queries)}: {query[:50]}...")
            result = self.translate(query, include_examples=include_examples)
            results.append(result)

        # Print summary
        successful = sum(1 for r in results if r.success)
        self._log(f"\n{'='*70}")
        self._log(f"BATCH COMPLETE: {successful}/{len(queries)} successful")
        self._log(f"{'='*70}")

        return results

    def explain(self, result: TranslationResult) -> str:
        """
        Generate human-readable explanation of a translation result.

        Args:
            result: TranslationResult to explain

        Returns:
            Formatted explanation string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("QUERY TRANSLATION EXPLANATION")
        lines.append("=" * 70)

        # User query
        lines.append(f"\nUser Query: \"{result.user_query}\"")

        # Intent analysis
        lines.append("\n[1] INTENT ANALYSIS (LLM)")
        lines.append("-" * 70)
        lines.append(f"Query Type: {result.intent_dict.get('query_type')}")
        lines.append(f"Resources Needed: {result.intent_dict.get('required_resources')}")
        lines.append(f"Complexity: {result.complexity_score}/5")

        if result.intent_dict.get('lemma'):
            lines.append(f"Target Lemma: {result.intent_dict['lemma']}")
        if result.intent_dict.get('semantic_relation'):
            lines.append(f"Semantic Relation: {result.intent_dict['semantic_relation']}")
        if result.intent_dict.get('definition_pattern'):
            lines.append(f"Definition Pattern: {result.intent_dict['definition_pattern']}")

        lines.append(f"\nReasoning: {result.intent_dict.get('reasoning', 'N/A')}")

        if result.intent_warnings:
            lines.append(f"\n[WARN] Intent Warnings:")
            for warning in result.intent_warnings:
                lines.append(f"  - {warning}")

        # Execution plan
        if result.execution_plan:
            lines.append("\n[2] EXECUTION PLAN (Orchestrator)")
            lines.append("-" * 70)
            lines.append(f"Total Steps: {result.total_steps}")
            lines.append("\nStep Sequence:")
            for i, step in enumerate(result.execution_plan.steps, 1):
                lines.append(f"  {i}. {step.tool_name}")
                lines.append(f"     -> {step.description}")
                if step.depends_on:
                    lines.append(f"     (depends on: {step.depends_on})")

            if not result.plan_valid:
                lines.append(f"\n[WARN] Plan Validation Errors:")
                for error in result.plan_errors:
                    lines.append(f"  - {error}")

        # Assembly
        lines.append("\n[3] SPARQL ASSEMBLY (Assembler)")
        lines.append("-" * 70)
        lines.append(f"Uses SERVICE clause: {result.uses_service}")
        lines.append(f"Has aggregation: {result.has_aggregation}")
        lines.append(f"\nVariable Mappings:")
        for var, source in result.variable_mappings.items():
            lines.append(f"  {var} <- {source}")

        if result.assembly_warnings:
            lines.append(f"\n[WARN] Assembly Warnings:")
            for warning in result.assembly_warnings:
                lines.append(f"  - {warning}")

        # Result
        lines.append("\n[4] RESULT")
        lines.append("-" * 70)
        if result.success:
            lines.append("[OK] Translation successful")
            lines.append(f"Processing time: {result.processing_time_ms:.1f}ms")
            lines.append(f"\nGenerated SPARQL:\n{result.sparql_query}")
        else:
            lines.append("[FAIL] Translation failed")
            lines.append(f"Error: {result.error_message}")

        return "\n".join(lines)

    def _log(self, message: str):
        """Internal logging helper"""
        if self.verbose:
            print(message)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_translator(llm_client, verbose: bool = True) -> Translator:
    """
    Convenience function to create a translator instance.

    Args:
        llm_client: LLM client for intent analysis
        verbose: Whether to print progress information

    Returns:
        Configured Translator instance
    """
    return Translator(llm_client, verbose=verbose)


def quick_translate(user_query: str, llm_client) -> str:
    """
    Quick translation without detailed metadata.

    Args:
        user_query: Natural language query
        llm_client: LLM client

    Returns:
        SPARQL query string
    """
    translator = Translator(llm_client, verbose=False)
    result = translator.translate(user_query)
    return result.sparql_query
