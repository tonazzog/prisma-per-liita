#!/usr/bin/env python3
"""
PRISMA per LiITA - Gradio Web UI
================================

A user-friendly web interface for the PRISMA NL2SPARQL system.
Shows the complete pipeline with intermediate results for transparency.

Usage:
    python -m app.gradio_app
    python -m app.gradio_app --provider anthropic --model claude-sonnet-4-20250514
    python -m app.gradio_app --provider ollama --model llama3 --share
"""

import argparse
import json
import traceback
from typing import Optional, Tuple

import gradio as gr
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prisma import Translator, TranslationResult, create_llm_client


# ============================================================================
# Configuration
# ============================================================================

LIITA_ENDPOINT = "https://liita.it/sparql"
DEFAULT_TIMEOUT = 30

# Example queries for the UI
EXAMPLE_QUERIES = [
    "What is the polarity of the word 'amore'?",
    "Find the hyponyms of 'colore' with their definitions",
    "What emotion is associated with 'felicità'?",
    "Find meronyms of 'corpo' with their Sicilian translation",
    "Translate 'casa' to Parmigiano dialect",
    "Find Sicilian nouns ending in 'ìa'",
    "How many nouns are in LiITA?",
    "Find words whose definition starts with 'uccello'",
    "What are the synonyms of 'veloce'?",
    "Find lemmas starting with 'infra'",
]


# ============================================================================
# Global State
# ============================================================================

translator: Optional[Translator] = None
current_provider: str = ""
current_model: str = ""


# ============================================================================
# Initialization
# ============================================================================

def init_translator(provider: str, model: Optional[str], api_key: Optional[str]) -> str:
    """Initialize the translator with the specified LLM provider."""
    global translator, current_provider, current_model

    try:
        # Create LLM client
        llm_client = create_llm_client(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.0
        )

        # Create translator
        translator = Translator(llm_client, verbose=False)

        current_provider = provider
        current_model = model or f"{provider} default"

        return f"Initialized with {provider} ({current_model})"

    except Exception as e:
        return f"Error initializing: {str(e)}"


# ============================================================================
# Translation Functions
# ============================================================================

def translate_query(question: str) -> Tuple[str, str, str, str, str]:
    """
    Translate a natural language question to SPARQL.

    Returns: (sparql, intent_info, plan_info, assembly_info, status_info)
    """
    if not translator:
        return "", "Error: Translator not initialized", "", "", "Please configure LLM settings first"

    if not question.strip():
        return "", "", "", "", "Please enter a question"

    try:
        # Perform translation
        result = translator.translate(question)

        # Format SPARQL output
        sparql = result.sparql_query if result.success else ""

        # Format Intent Analysis section
        intent_info = format_intent_analysis(result)

        # Format Execution Plan section
        plan_info = format_execution_plan(result)

        # Format Assembly section
        assembly_info = format_assembly_info(result)

        # Format Status section
        status_info = format_status_info(result)

        return sparql, intent_info, plan_info, assembly_info, status_info

    except Exception as e:
        error_msg = f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return "", error_msg, "", "", "Translation failed"


def format_intent_analysis(result: TranslationResult) -> str:
    """Format the intent analysis section."""
    lines = []

    intent = result.intent_dict

    if not intent:
        return "No intent analysis available"

    # Header with query type
    query_type = intent.get('query_type', 'unknown')
    lines.append(f"### Query Type: `{query_type}`\n")

    # Key extracted information
    lines.append("**Extracted Information:**\n")

    if intent.get('lemma'):
        lines.append(f"- **Lemma:** `{intent['lemma']}`")

    if intent.get('lemma_b'):
        lines.append(f"- **Second Lemma:** `{intent['lemma_b']}`")

    if intent.get('pos'):
        lines.append(f"- **Part of Speech:** {intent['pos']}")

    if intent.get('semantic_relation'):
        lines.append(f"- **Semantic Relation:** {intent['semantic_relation']}")

    if intent.get('definition_pattern'):
        pattern_type = intent.get('pattern_type', 'contains')
        lines.append(f"- **Definition Pattern:** `{intent['definition_pattern']}` ({pattern_type})")

    if intent.get('written_form_pattern'):
        lines.append(f"- **Written Form Pattern:** `{intent['written_form_pattern']}`")

    # Resources
    resources = intent.get('required_resources', [])
    if resources:
        resource_badges = " ".join([f"`{r}`" for r in resources])
        lines.append(f"\n**Required Resources:** {resource_badges}")

    # Complexity
    complexity = intent.get('complexity_score', 1)
    complexity_bar = "█" * complexity + "░" * (5 - complexity)
    lines.append(f"\n**Complexity:** [{complexity_bar}] {complexity}/5")

    # Aggregation
    if intent.get('aggregation'):
        agg = intent['aggregation']
        agg_type = agg.get('type', 'none')
        lines.append(f"\n**Aggregation:** {agg_type}")
        if agg.get('group_by_vars'):
            lines.append(f"  - Group by: {', '.join(agg['group_by_vars'])}")

    # Reasoning
    if intent.get('reasoning'):
        lines.append(f"\n**LLM Reasoning:**\n> {intent['reasoning']}")

    # Warnings
    if result.intent_warnings:
        lines.append(f"\n**Warnings:**")
        for warning in result.intent_warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def format_execution_plan(result: TranslationResult) -> str:
    """Format the execution plan section."""
    lines = []

    if not result.execution_plan:
        return "No execution plan available"

    plan = result.execution_plan

    lines.append(f"### Steps: {len(plan.steps)}\n")

    # Show each step
    for step in plan.steps:
        # Step header
        lines.append(f"**Step {step.step_number}: `{step.tool_name}`**")
        lines.append(f"> {step.description}")

        # Parameters (formatted nicely)
        if step.parameters:
            params_str = ", ".join([f"{k}=`{v}`" for k, v in step.parameters.items() if v is not None])
            if params_str:
                lines.append(f"- Parameters: {params_str}")

        # Dependencies
        if step.depends_on:
            deps = ", ".join([f"Step {d}" for d in step.depends_on])
            lines.append(f"- Depends on: {deps}")

        lines.append("")  # Blank line between steps

    # Variable flow
    if plan.variable_flow:
        lines.append("**Variable Flow:**")
        for var, source in plan.variable_flow.items():
            lines.append(f"- `{var}` <- {source}")

    # Plan validation
    if not result.plan_valid:
        lines.append("\n**Validation Errors:**")
        for error in result.plan_errors:
            lines.append(f"- {error}")

    return "\n".join(lines)


def format_assembly_info(result: TranslationResult) -> str:
    """Format the assembly information section."""
    lines = []

    if not result.sparql_query:
        return "No assembly information available"

    # Metadata
    lines.append("### Assembly Metadata\n")
    lines.append(f"- **Uses SERVICE clause:** {'Yes' if result.uses_service else 'No'}")
    lines.append(f"- **Has aggregation:** {'Yes' if result.has_aggregation else 'No'}")
    lines.append(f"- **Total steps:** {result.total_steps}")

    # Variable mappings
    if result.variable_mappings:
        lines.append("\n**Variable Mappings:**")
        for var, source in result.variable_mappings.items():
            lines.append(f"- `?{var}` <- {source}")

    # Assembly warnings
    if result.assembly_warnings:
        lines.append("\n**Warnings:**")
        for warning in result.assembly_warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def format_status_info(result: TranslationResult) -> str:
    """Format the status information section."""
    if result.success:
        return f"Translation successful in {result.processing_time_ms:.1f}ms"
    else:
        return f"Translation failed: {result.error_message}"


# ============================================================================
# SPARQL Execution
# ============================================================================

def execute_sparql(sparql: str, limit: int = 20) -> str:
    """Execute a SPARQL query against the LiITA endpoint."""
    if not sparql.strip():
        return "Please enter a SPARQL query"

    try:
        # Add LIMIT if not present
        if "LIMIT" not in sparql.upper():
            sparql = sparql.rstrip().rstrip(';') + f"\nLIMIT {limit}"

        # Execute query
        response = requests.post(
            LIITA_ENDPOINT,
            data={"query": sparql},
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            timeout=DEFAULT_TIMEOUT
        )

        if response.status_code != 200:
            return f"**Error:** HTTP {response.status_code}\n\n```\n{response.text}\n```"

        # Parse results
        data = response.json()

        # Extract variables and bindings
        variables = data.get('head', {}).get('vars', [])
        bindings = data.get('results', {}).get('bindings', [])

        # Format output
        output = f"### Results: {len(bindings)} rows\n\n"
        output += f"**Variables:** {', '.join(variables)}\n\n"

        if bindings:
            # Create markdown table
            output += "| " + " | ".join(variables) + " |\n"
            output += "| " + " | ".join(["---"] * len(variables)) + " |\n"

            for row in bindings[:limit]:
                values = []
                for var in variables:
                    if var in row:
                        val = row[var].get('value', '')
                        # Truncate long values
                        if len(val) > 50:
                            val = val[:47] + "..."
                        values.append(val)
                    else:
                        values.append("")
                output += "| " + " | ".join(values) + " |\n"

            if len(bindings) > limit:
                output += f"\n*Showing first {limit} of {len(bindings)} results*"
        else:
            output += "*No results found*"

        return output

    except requests.exceptions.Timeout:
        return "**Error:** Query timed out"
    except requests.exceptions.RequestException as e:
        return f"**Error:** Request failed: {str(e)}"
    except json.JSONDecodeError as e:
        return f"**Error:** Invalid JSON response: {str(e)}"
    except Exception as e:
        return f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


# ============================================================================
# Provider Configuration
# ============================================================================

def update_provider(provider: str, model: str, api_key: str) -> str:
    """Update the LLM provider configuration."""
    if not provider:
        return "Please select a provider"

    # Use provided model or default
    model_to_use = model.strip() if model.strip() else None

    # Initialize translator
    status = init_translator(provider, model_to_use, api_key.strip() if api_key.strip() else None)

    return status


# ============================================================================
# Gradio UI
# ============================================================================

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    with gr.Blocks(title="PRISMA per LiITA - Pattern-Based NL2SPARQL") as app:
        gr.Markdown("""
        # PRISMA per LiITA

        **P**attern-based **R**ules for **I**nterrogating Li**ITA** **S**PARQL with **M**ultiple-resource **A**ssembly

        Translate natural language questions into SPARQL queries for the
        [LiITA (Linking Italian)](https://liita.it) linguistic knowledge base.

        **Pipeline:** Intent Analysis (LLM) → Execution Planning (Rules) → Pattern Assembly → SPARQL
        """)

        with gr.Tabs():
            # ================================================================
            # Tab 1: Translate
            # ================================================================
            with gr.TabItem("Translate"):
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Natural Language Question",
                        placeholder="e.g., What is the polarity of the word 'amore'?",
                        lines=2,
                        scale=4,
                    )
                    translate_btn = gr.Button("Translate", variant="primary", scale=1)

                # Status bar
                status_output = gr.Markdown(label="Status")

                with gr.Row():
                    # Left column: SPARQL output
                    with gr.Column(scale=2):
                        sparql_output = gr.Code(
                            label="Generated SPARQL",
                            language="sparql",
                            lines=15,
                        )
                        with gr.Row():
                            copy_btn = gr.Button("Copy to Execute Tab", size="sm")

                    # Right column: Pipeline details
                    with gr.Column(scale=1):
                        with gr.Accordion("1. Intent Analysis (LLM)", open=True):
                            intent_output = gr.Markdown()

                        with gr.Accordion("2. Execution Plan (Rules)", open=True):
                            plan_output = gr.Markdown()

                        with gr.Accordion("3. Pattern Assembly", open=False):
                            assembly_output = gr.Markdown()

                # Wire up translate button
                translate_btn.click(
                    translate_query,
                    inputs=[question_input],
                    outputs=[sparql_output, intent_output, plan_output, assembly_output, status_output],
                )

                # Also translate on Enter
                question_input.submit(
                    translate_query,
                    inputs=[question_input],
                    outputs=[sparql_output, intent_output, plan_output, assembly_output, status_output],
                )

                # Examples
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUERIES],
                    inputs=[question_input],
                    label="Example Queries",
                )

            # ================================================================
            # Tab 2: Execute SPARQL
            # ================================================================
            with gr.TabItem("Execute SPARQL"):
                gr.Markdown("""
                ### Execute SPARQL Query

                Run SPARQL queries directly against the LiITA endpoint.
                You can paste queries from the Translate tab or write your own.
                """)

                exec_sparql_input = gr.Code(
                    label="SPARQL Query",
                    language="sparql",
                    lines=12,
                    value="""PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX lexinfo: <http://www.lexinfo.net/ontology/3.0/lexinfo#>

SELECT ?lemma ?writtenRep
WHERE {
    ?lemma a ontolex:LexicalEntry ;
           ontolex:canonicalForm ?form ;
           lexinfo:partOfSpeech lexinfo:noun .
    ?form ontolex:writtenRep ?writtenRep .
}
LIMIT 10""",
                )

                with gr.Row():
                    exec_limit = gr.Slider(
                        minimum=5, maximum=100, value=20, step=5,
                        label="Result Limit",
                    )
                    exec_btn = gr.Button("Execute Query", variant="primary")

                exec_output = gr.Markdown(label="Results")

                exec_btn.click(
                    execute_sparql,
                    inputs=[exec_sparql_input, exec_limit],
                    outputs=[exec_output],
                )

                # Store reference for copy button
                copy_btn.click(
                    lambda x: x,
                    inputs=[sparql_output],
                    outputs=[exec_sparql_input],
                )

            # ================================================================
            # Tab 3: Settings
            # ================================================================
            with gr.TabItem("Settings"):
                gr.Markdown("""
                ### LLM Provider Configuration

                Configure the LLM provider used for intent analysis.
                The LLM is only used for classifying queries - SPARQL generation is rule-based.
                """)

                with gr.Row():
                    provider_dropdown = gr.Dropdown(
                        choices=["mistral", "anthropic", "openai", "gemini", "ollama"],
                        value="mistral",
                        label="Provider",
                    )
                    model_input = gr.Textbox(
                        label="Model (optional, uses default if empty)",
                        placeholder="e.g., claude-sonnet-4-20250514, gpt-4o, mistral-large-latest",
                    )

                api_key_input = gr.Textbox(
                    label="API Key (optional, uses environment variable if empty)",
                    placeholder="Your API key (not needed for Ollama)",
                    type="password",
                )

                config_btn = gr.Button("Apply Configuration", variant="primary")
                config_status = gr.Markdown(label="Configuration Status")

                config_btn.click(
                    update_provider,
                    inputs=[provider_dropdown, model_input, api_key_input],
                    outputs=[config_status],
                )

                gr.Markdown("""
                ---

                **Default Models:**
                - Mistral: `mistral-large-latest`
                - Anthropic: `claude-sonnet-4-20250514`
                - OpenAI: `gpt-4o`
                - Gemini: `gemini-1.5-pro`
                - Ollama: `llama3.1`

                **Environment Variables:**
                - `MISTRAL_API_KEY`
                - `ANTHROPIC_API_KEY`
                - `OPENAI_API_KEY`
                - `GOOGLE_API_KEY`
                """)

        gr.Markdown("""
        ---
        **PRISMA per LiITA** - Pattern-based Rules for Intent-driven SPARQL with Multiple-resource Assembly |
        [GitHub](https://github.com/yourusername/prisma-per-liita)
        """)

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISMA per LiITA - Gradio Web UI")
    parser.add_argument(
        "--provider", "-p",
        default="mistral",
        choices=["openai", "anthropic", "mistral", "gemini", "ollama"],
        help="LLM provider (default: mistral)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (uses environment variable if not specified)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PRISMA per LiITA - Gradio Web UI")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print()

    # Initialize translator
    print("Initializing translator...")
    status = init_translator(args.provider, args.model, args.api_key)
    print(status)
    print()

    # Create and launch UI
    print("Starting web UI...")
    app = create_ui()
    app.launch(
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
