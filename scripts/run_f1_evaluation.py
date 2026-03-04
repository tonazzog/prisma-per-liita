#!/usr/bin/env python3
"""Run F1 evaluation on the test dataset using PRISMA.

Translates every NL question in the dataset with PRISMA's Translator,
executes both the gold and predicted SPARQL queries against the LiITA
endpoint, and computes F1 on the result sets.

The output JSON file contains aggregate metrics AND all generated queries
for inspection and comparison with nl2sparql results.

Usage:
    python scripts/run_f1_evaluation.py --provider mistral --strip-limit
    python scripts/run_f1_evaluation.py --provider anthropic --model claude-haiku-4-5-20251001 --strip-limit
    python scripts/run_f1_evaluation.py --provider mistral --no-prefetch --delay 1.1
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Add the project root to sys.path so we can import prisma and shared
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.f1_evaluator import F1Evaluator, save_f1_report


# ---------------------------------------------------------------------------
# PRISMA adapter
# ---------------------------------------------------------------------------

class PrismaTranslatorAdapter:
    """Adapts PRISMA's Translator to the interface expected by F1Evaluator.

    F1Evaluator calls translator.translate(question) and reads .sparql on
    the result.  PRISMA's TranslationResult uses .sparql_query instead, so
    this thin wrapper bridges the two interfaces.
    """

    def __init__(self, translator):
        self._translator = translator

    def translate(self, question: str, **kwargs):
        result = self._translator.translate(question)
        if not result.success:
            # Raise so F1Evaluator stores the actual error message (not a generic placeholder).
            raise RuntimeError(result.error_message or "Translation failed (no error message)")
        return SimpleNamespace(sparql=result.sparql_query)


# ---------------------------------------------------------------------------
# Rate limiter (ported from nl2sparql/scripts/run_f1_evaluation.py)
# ---------------------------------------------------------------------------

class _RateLimitedTranslator:
    """Wraps any translator and enforces rate limits between calls.

    Handles two independent constraints:
      - RPS (requests per second): minimum wall-clock gap between calls
      - TPM (tokens per minute): sliding-window token budget

    The effective delay is max(rps_delay, tpm_delay) so both limits are
    respected simultaneously. Token counting uses a character-based estimate
    (chars / 4) for the question plus a fixed overhead for the system prompt,
    few-shot examples, and generated output.
    """

    # Conservative estimate of tokens that don't vary per question:
    # system prompt + constraints (~8 000) + 5 examples (~1 750) + output (~500)
    _OVERHEAD_TOKENS = 10_250

    def __init__(
        self,
        translator,
        delay: float,
        tpm_limit: int = 0,
        tokens_per_call: int = 0,
    ):
        """
        Args:
            translator:      The wrapped translator object.
            delay:           Minimum seconds between calls (RPS-based). 0 = no RPS cap.
            tpm_limit:       Tokens-per-minute budget. 0 = no TPM cap.
            tokens_per_call: Override the per-call token estimate used for TPM
                             accounting. 0 = auto (overhead + question chars/4).
        """
        self._translator = translator
        self._delay = delay
        self._tpm_limit = tpm_limit
        self._tokens_per_call_override = tokens_per_call
        self._last_call: float = 0.0
        # Sliding window: list of (monotonic_time, tokens) for the last 60 s
        self._window: list[tuple[float, int]] = []

    def _estimate_tokens(self, question: str) -> int:
        if self._tokens_per_call_override:
            return self._tokens_per_call_override
        return self._OVERHEAD_TOKENS + max(1, len(question) // 4)

    def _tpm_wait(self, tokens: int) -> float:
        """Return seconds to sleep so that adding `tokens` stays under TPM limit."""
        if not self._tpm_limit:
            return 0.0

        now = time.monotonic()
        cutoff = now - 60.0
        # Drop entries older than 60 s
        self._window = [(t, tok) for t, tok in self._window if t > cutoff]

        used = sum(tok for _, tok in self._window)
        headroom = self._tpm_limit - used

        if tokens <= headroom:
            return 0.0

        # Find the earliest entry whose removal would free enough space
        cumulative = 0
        for ts, tok in sorted(self._window):
            cumulative += tok
            if used - cumulative + self._tpm_limit >= tokens:
                # Sleep until that entry expires
                return max(0.0, (ts + 60.0) - now)

        # Worst case: wait until the whole window clears
        if self._window:
            oldest = min(ts for ts, _ in self._window)
            return max(0.0, (oldest + 60.0) - now)
        return 0.0

    def translate(self, question: str):
        tokens = self._estimate_tokens(question)

        # RPS-based delay
        rps_wait = max(0.0, self._delay - (time.monotonic() - self._last_call))

        # TPM-based delay
        tpm_wait = self._tpm_wait(tokens)

        wait = max(rps_wait, tpm_wait)
        if wait > 0:
            time.sleep(wait)

        self._last_call = time.monotonic()
        result = self._translator.translate(question)

        # Record this call in the sliding window
        if self._tpm_limit:
            self._window.append((time.monotonic(), tokens))

        return result


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_test_dataset(path: str = None) -> dict:
    """Load the test dataset JSON file.

    Args:
        path: Path to test dataset JSON. If None, uses data/test_dataset.json
              relative to the project root.

    Returns:
        Loaded dataset dict with "test_cases" key.
    """
    if path is None:
        path = Path(__file__).parent.parent / "data" / "test_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report merging (for --rerun-errors)
# ---------------------------------------------------------------------------

def _load_error_ids(report_path: Path, error_filter: str = "") -> set[int]:
    """Return the set of test_ids that have a predicted_error in a report.

    Args:
        report_path:  Path to the existing JSON report.
        error_filter: Optional substring to match against the predicted_error string.
                      If empty, all cases with any predicted_error are returned.
    """
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        r["test_id"] for r in data["results"]
        if r.get("predicted_error") and (not error_filter or error_filter in r["predicted_error"])
    }


def _merge_report(
    existing_path: Path,
    new_results,           # list[F1Result] from the re-run
    test_data: dict,       # full dataset (needed for category/pattern metadata)
) -> dict:
    """Merge new F1Results into an existing report dict and recalculate all stats.

    For every test_id present in new_results, the old entry is replaced.
    All other entries are kept unchanged. Summary metrics are recomputed from
    the merged result list.
    """
    from evaluation.f1_evaluator import F1Result

    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)

    tc_by_id = {tc["id"]: tc for tc in test_data["test_cases"]}
    new_by_id = {r.test_id: r for r in new_results}

    # Rebuild result list: replace updated entries, keep everything else
    merged: list[F1Result] = []
    for old in existing["results"]:
        tid = old["test_id"]
        if tid in new_by_id:
            merged.append(new_by_id[tid])
        else:
            merged.append(F1Result(
                test_id=tid,
                precision=old["precision"],
                recall=old["recall"],
                f1=old["f1"],
                gold_count=old["gold_count"],
                predicted_count=old["predicted_count"],
                true_positives=old["true_positives"],
                aggregate_score=old.get("aggregate_score"),
                aggregate_details=old.get("aggregate_details", {}),
                variable_mapping=old.get("variable_mapping", {}),
                gold_error=old.get("gold_error"),
                predicted_error=old.get("predicted_error"),
                predicted_sparql=old.get("predicted_sparql"),
            ))

    # Recalculate aggregate metrics
    n = len(merged)
    avg_precision = sum(r.precision for r in merged) / n if n else 0.0
    avg_recall    = sum(r.recall    for r in merged) / n if n else 0.0
    avg_f1        = sum(r.f1        for r in merged) / n if n else 0.0

    # Recalculate by_category and by_pattern from merged results + dataset metadata
    f1_by_cat: dict[str, list[float]] = {}
    f1_by_pat: dict[str, list[float]] = {}
    for r in merged:
        if r.gold_error:
            continue
        tc = tc_by_id.get(r.test_id, {})
        cat = tc.get("category", "unknown")
        f1_by_cat.setdefault(cat, []).append(r.f1)
        for pat in tc.get("patterns", []):
            f1_by_pat.setdefault(pat, []).append(r.f1)

    cat_avgs = [sum(v) / len(v) for v in f1_by_cat.values()]
    macro_f1 = sum(cat_avgs) / len(cat_avgs) if cat_avgs else 0.0

    cat_report = {c: {"avg_f1": sum(v) / len(v), "count": len(v)} for c, v in f1_by_cat.items()}
    pat_report = {p: {"avg_f1": sum(v) / len(v), "count": len(v)} for p, v in f1_by_pat.items()}

    # Preserve the original skipped count (gold failures don't change between runs)
    total_skipped = existing["summary"]["total_skipped"]

    return {
        "summary": {
            "total_evaluated": n,
            "total_skipped": total_skipped,
            "avg_precision": avg_precision,
            "avg_recall":    avg_recall,
            "avg_f1":        avg_f1,
            "macro_f1":      macro_f1,
        },
        "by_category": cat_report,
        "by_pattern":  pat_report,
        "results": [
            {
                "test_id":          r.test_id,
                "f1":               r.f1,
                "precision":        r.precision,
                "recall":           r.recall,
                "gold_count":       r.gold_count,
                "predicted_count":  r.predicted_count,
                "true_positives":   r.true_positives,
                "aggregate_score":  r.aggregate_score,
                "aggregate_details": r.aggregate_details,
                "variable_mapping": r.variable_mapping,
                "predicted_sparql": r.predicted_sparql,
                "gold_error":       r.gold_error,
                "predicted_error":  r.predicted_error,
            }
            for r in merged
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F1 evaluation for PRISMA: translate NL questions and compare answer sets"
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="LLM provider (mistral / anthropic / openai / gemini / ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (uses provider default if omitted)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (uses environment variable if omitted)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to test dataset JSON (default: data/test_dataset.json)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["it", "en"],
        help="NL question language to use (default: en; the dataset only has nl_en)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: reports/f1_report_prisma_<provider>_<model>.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SPARQL endpoint timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds between translation calls (RPS cap). "
            "E.g. 1.1 for Mistral free tier (1 req/s). Default: 0."
        ),
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=0,
        dest="tpm_limit",
        help=(
            "Tokens-per-minute budget (TPM cap). The script will automatically "
            "pause when the sliding-window token count approaches this limit. "
            "E.g. --tpm-limit 100000 for a 100K TPM free tier. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--tokens-per-call",
        type=int,
        default=0,
        dest="tokens_per_call",
        help=(
            "Override the per-call token estimate used for TPM accounting. "
            "Default: 0 (auto: ~10 250 overhead + question length / 4)."
        ),
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable gold query pre-fetching (saves memory, slower for repeated runs)",
    )
    parser.add_argument(
        "--no-strip-limit",
        action="store_false",
        dest="strip_limit",
        help=(
            "Disable stripping of LIMIT and OFFSET clauses from both gold and "
            "predicted queries before executing them. By default LIMIT/OFFSET are "
            "always removed to ensure fair result-set comparison."
        ),
    )
    parser.set_defaults(strip_limit=True)
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        dest="rerun_errors",
        help=(
            "Only re-evaluate test cases that have a predicted_error in the existing "
            "report (--output path). New results are merged back in and all summary "
            "statistics are recalculated. Use --error-filter to target a specific error."
        ),
    )
    parser.add_argument(
        "--error-filter",
        default="",
        dest="error_filter",
        metavar="SUBSTRING",
        help=(
            "When used with --rerun-errors, only re-run cases whose predicted_error "
            "contains this substring. Default: re-run all cases with any error. "
            "Example: --error-filter \"NoneType\" or --error-filter \"529\"."
        ),
    )

    args = parser.parse_args()

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        model_tag = (args.model or "default").replace("/", "-").replace(":", "-")
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"f1_report_prisma_{args.provider}_{model_tag}.json"

    # Load dataset
    print("Loading dataset...")
    test_data = load_test_dataset(args.dataset)
    total = len(test_data["test_cases"])
    print(f"  {total} test cases loaded")
    if args.dataset:
        print(f"  Source: {args.dataset}")

    # --rerun-errors: filter dataset to only previously-failed cases
    if args.rerun_errors:
        if not output_path.exists():
            print(f"\nError: --rerun-errors requires an existing report at {output_path}")
            print("Run a full evaluation first, then use --rerun-errors to patch it.")
            sys.exit(1)
        error_ids = _load_error_ids(output_path, error_filter=args.error_filter)
        filter_desc = f"containing '{args.error_filter}'" if args.error_filter else "(any error)"
        if not error_ids:
            print(f"\nNo cases with predicted_error {filter_desc} found in existing report — nothing to re-run.")
            sys.exit(0)
        test_data["test_cases"] = [tc for tc in test_data["test_cases"] if tc["id"] in error_ids]
        total = len(test_data["test_cases"])
        print(f"\n  --rerun-errors: {total} cases with predicted_error {filter_desc} selected")
        print(f"  IDs: {sorted(error_ids)}")

    # Build PRISMA translator
    print(f"\nInitialising PRISMA translator...")
    print(f"  Provider : {args.provider}")
    print(f"  Model    : {args.model or '(provider default)'}")
    print(f"  Language : {args.language}")

    try:
        from shared.llm import create_llm_client
        from prisma.translator import Translator

        llm_client = create_llm_client(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
        )
        prisma_translator = Translator(llm_client, verbose=False)
        translator = PrismaTranslatorAdapter(prisma_translator)

    except Exception as e:
        print(f"\nError initialising PRISMA translator: {e}")
        sys.exit(1)

    # Build evaluator
    print(f"  Strip LIMIT: {'enabled' if args.strip_limit else 'disabled'} (LIMIT/OFFSET {'removed from' if args.strip_limit else 'kept in'} both queries)")
    evaluator = F1Evaluator(
        timeout=args.timeout,
        cache_gold_results=not args.no_prefetch,
        strip_limit=args.strip_limit,
    )

    # Pre-fetch gold results (runs all gold queries once up front)
    if not args.no_prefetch:
        print(f"\nPre-fetching gold query results ({total} queries)...")
        evaluator.prefetch_gold_results(test_data)
        cached = len(evaluator._gold_cache)
        skipped = total - cached
        print(f"  Cached: {cached}  |  Failed/skipped: {skipped}")

    # Wrap translator with rate limiter if requested
    if args.delay > 0 or args.tpm_limit > 0:
        translator = _RateLimitedTranslator(
            translator,
            delay=args.delay,
            tpm_limit=args.tpm_limit,
            tokens_per_call=args.tokens_per_call,
        )
        est = args.tokens_per_call or (
            _RateLimitedTranslator._OVERHEAD_TOKENS + 40   # ~40 tokens avg question
        )
        print(f"\nRate limiting:")
        if args.delay > 0:
            print(f"  RPS cap : {args.delay}s between calls  "
                  f"(max {60/args.delay:.0f} calls/min)")
        if args.tpm_limit > 0:
            max_calls_tpm = args.tpm_limit / est
            safe_delay = 60.0 / max_calls_tpm
            print(f"  TPM cap : {args.tpm_limit:,} tokens/min  "
                  f"(~{est:,} tokens/call → max {max_calls_tpm:.1f} calls/min, "
                  f"effective delay ≥ {safe_delay:.1f}s)")
        effective = max(args.delay, (60.0 / (args.tpm_limit / est)) if args.tpm_limit else 0)
        total_est = len(test_data["test_cases"]) * effective
        print(f"  Estimated total run time: ~{total_est/60:.1f} min")

    # Run evaluation
    print(f"\nTranslating and evaluating ({total} questions)...")
    report = evaluator.evaluate_dataset(
        test_data=test_data,
        translator=translator,
        language=args.language,
    )

    # Save (merge into existing report when --rerun-errors, otherwise write fresh)
    if args.rerun_errors:
        full_test_data = load_test_dataset(args.dataset)  # need full dataset for category metadata
        merged = _merge_report(output_path, report.results, full_test_data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        # Reflect merged stats in the summary printed below
        report.avg_f1        = merged["summary"]["avg_f1"]
        report.avg_precision = merged["summary"]["avg_precision"]
        report.avg_recall    = merged["summary"]["avg_recall"]
        report.macro_f1      = merged["summary"]["macro_f1"]
        report.total_evaluated = merged["summary"]["total_evaluated"]
        report.total_skipped   = merged["summary"]["total_skipped"]
        report.f1_by_category  = merged["by_category"]
        report.f1_by_pattern   = merged["by_pattern"]
        report.results         = report.results  # keep the re-run subset for score distribution
    else:
        save_f1_report(report, str(output_path))

    # Summary
    rerun_tag = "  [re-run merged]" if args.rerun_errors else ""
    print(f"\n{'='*55}")
    print(f"F1 EVALUATION RESULTS  —  PRISMA / {args.provider} / {args.model or 'default'}{rerun_tag}")
    print(f"{'='*55}")
    print(f"  Evaluated : {report.total_evaluated}")
    print(f"  Skipped   : {report.total_skipped}  (gold query failed)")
    print(f"  Avg F1    : {report.avg_f1:.4f}")
    print(f"  Macro F1  : {report.macro_f1:.4f}")
    print(f"  Precision : {report.avg_precision:.4f}")
    print(f"  Recall    : {report.avg_recall:.4f}")

    if report.f1_by_category:
        print(f"\n  By category:")
        for cat, stats in sorted(report.f1_by_category.items()):
            print(f"    {cat:<25} avg F1={stats['avg_f1']:.4f}  (n={stats['count']})")

    if report.f1_by_pattern:
        print(f"\n  By pattern (top 10):")
        sorted_pats = sorted(
            report.f1_by_pattern.items(),
            key=lambda x: -x[1]["avg_f1"],
        )[:10]
        for pat, stats in sorted_pats:
            print(f"    {pat:<28} avg F1={stats['avg_f1']:.4f}  (n={stats['count']})")

    # Distribution of F1 scores
    if report.results:
        perfect = sum(1 for r in report.results if r.f1 == 1.0)
        zeros   = sum(1 for r in report.results if r.f1 == 0.0)
        errors  = sum(1 for r in report.results if r.predicted_error)
        print(f"\n  Score distribution:")
        print(f"    F1 = 1.00 (perfect): {perfect}")
        print(f"    F1 = 0.00          : {zeros}")
        print(f"    Translation errors : {errors}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
