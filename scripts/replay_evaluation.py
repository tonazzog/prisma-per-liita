#!/usr/bin/env python3
"""Replay F1 evaluation using predicted SPARQLs stored in an existing report.

Re-executes both gold (from the updated test dataset) and predicted (from the
existing report) queries against the endpoint and recomputes F1 scores.

Useful when:
  - The gold dataset has been fixed but the LLM predictions don't need to change
  - You want to assess gold vs predicted query quality for failed cases

Usage:
    python scripts/replay_evaluation.py reports/f1_report_prisma_ollama_glm-5-cloud.json
    python scripts/replay_evaluation.py reports/f1_report_prisma_ollama_glm-5-cloud.json --output reports/f1_report_glm5_replayed.json
    python scripts/replay_evaluation.py reports/f1_report_prisma_ollama_glm-5-cloud.json --show-failures
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.f1_evaluator import (
    F1Evaluator,
    execute_query_full,
    strip_limit_offset,
    save_f1_report,
)


def load_dataset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {t["id"]: t for t in data["test_cases"]}


def load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def replay(
    report_path: Path,
    dataset_path: Path,
    output_path: Path,
    strip_limit: bool = True,
    show_failures: bool = False,
    show_all: bool = False,
    timeout: int = 60,
):
    report = load_report(report_path)
    dataset = load_dataset(dataset_path)
    evaluator = F1Evaluator(strip_limit=strip_limit)

    old_results = {r["test_id"]: r for r in report["results"]}
    test_ids = sorted(old_results.keys())

    new_results = []
    improved = []
    regressed = []
    unchanged_zero = []
    unchanged_perfect = []

    print(f"\nReplaying {len(test_ids)} test cases from {report_path.name}")
    print(f"Gold dataset: {dataset_path}")
    print(f"strip_limit={strip_limit}\n")

    for i, tid in enumerate(test_ids, 1):
        old = old_results[tid]
        test = dataset.get(tid)
        if test is None:
            print(f"  [WARN] test_id {tid} not found in dataset, skipping")
            continue

        gold_sparql = test["sparql"]
        predicted_sparql = old.get("predicted_sparql") or ""
        answer_vars = test.get("answer_variables", {})
        old_f1 = old.get("f1", 0.0)

        if not predicted_sparql:
            # No prediction stored — keep as error
            result = type("R", (), {
                "test_id": tid, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "gold_count": 0, "predicted_count": 0, "true_positives": 0,
                "aggregate_score": None, "aggregate_details": {},
                "variable_mapping": {}, "gold_error": None,
                "predicted_error": old.get("predicted_error", "No predicted SPARQL"),
            })()
        else:
            try:
                result = evaluator.evaluate_single(
                    gold_sparql, predicted_sparql, answer_vars, test_id=tid
                )
            except Exception as e:
                result = type("R", (), {
                    "test_id": tid, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                    "gold_count": 0, "predicted_count": 0, "true_positives": 0,
                    "aggregate_score": None, "aggregate_details": {},
                    "variable_mapping": {}, "gold_error": None,
                    "predicted_error": str(e),
                })()

        new_f1 = result.f1
        delta = new_f1 - old_f1

        # Track changes
        if abs(delta) > 0.001:
            if delta > 0:
                improved.append((tid, old_f1, new_f1, delta))
            else:
                regressed.append((tid, old_f1, new_f1, delta))
        elif new_f1 == 0.0:
            unchanged_zero.append(tid)
        elif new_f1 >= 0.999:
            unchanged_perfect.append(tid)

        # Build result dict
        pred_error = getattr(result, "predicted_error", None)
        gold_error = getattr(result, "gold_error", None)
        r_dict = {
            "test_id": tid,
            "category": test.get("category", "unknown"),
            "nl_en": test.get("nl_en", ""),
            "f1": new_f1,
            "f1_old": old_f1,
            "precision": result.precision,
            "recall": result.recall,
            "gold_count": result.gold_count,
            "predicted_count": result.predicted_count,
            "true_positives": result.true_positives,
            "aggregate_score": result.aggregate_score,
            "aggregate_details": result.aggregate_details,
            "variable_mapping": result.variable_mapping,
            "gold_sparql": gold_sparql,
            "predicted_sparql": predicted_sparql,
            "gold_error": gold_error,
            "predicted_error": pred_error,
        }
        new_results.append(r_dict)

        status = "✓" if new_f1 >= 0.999 else ("!" if pred_error else "✗")
        delta_str = f" ({delta:+.3f})" if abs(delta) > 0.001 else ""
        print(f"  [{i:3d}/{len(test_ids)}] ID {tid}: F1={new_f1:.3f}{delta_str} {status}")

        if (show_failures and new_f1 == 0 and not pred_error) or show_all:
            _print_comparison(tid, test, gold_sparql, predicted_sparql,
                              result, old_f1, evaluator.endpoint, strip_limit)

    # Compute summary
    f1_scores = [r["f1"] for r in new_results]
    n = len(f1_scores)
    avg_f1 = sum(f1_scores) / n if n else 0.0
    perfect = sum(1 for f in f1_scores if f >= 0.999)
    zero = sum(1 for f in f1_scores if f == 0.0)
    errors = sum(1 for r in new_results if r.get("predicted_error"))

    summary = {
        "source_report": str(report_path),
        "total_evaluated": n,
        "avg_f1": avg_f1,
        "perfect_count": perfect,
        "zero_count": zero,
        "error_count": errors,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
    }

    # Save
    output = {
        "summary": summary,
        "results": new_results,
        "changes": {
            "improved": [{"id": t, "old": o, "new": nw, "delta": d} for t, o, nw, d in improved],
            "regressed": [{"id": t, "old": o, "new": nw, "delta": d} for t, o, nw, d in regressed],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"REPLAY RESULTS — {report_path.name}")
    print(f"{'='*60}")
    print(f"  Total evaluated : {n}")
    print(f"  Avg F1          : {avg_f1:.4f}  (was {report['summary']['avg_f1']:.4f})")
    print(f"  Perfect (F1=1)  : {perfect}")
    print(f"  Zero (F1=0)     : {zero}")
    print(f"  Errors          : {errors}")
    print(f"  Improved        : {len(improved)}")
    print(f"  Regressed       : {len(regressed)}")
    if improved:
        print(f"\n  Top improvements:")
        for tid, o, nw, d in sorted(improved, key=lambda x: -x[3])[:10]:
            print(f"    ID {tid}: {o:.3f} -> {nw:.3f} ({d:+.3f})")
    if regressed:
        print(f"\n  Regressions:")
        for tid, o, nw, d in sorted(regressed, key=lambda x: x[3])[:10]:
            print(f"    ID {tid}: {o:.3f} -> {nw:.3f} ({d:+.3f})")
    print(f"\n  Report saved to: {output_path}")


def _print_comparison(tid, test, gold_sparql, pred_sparql,
                       result, old_f1, endpoint, strip_limit):
    """Print side-by-side gold vs predicted comparison for a failed case."""
    print(f"\n  {'─'*70}")
    print(f"  ID {tid}: {test.get('nl_en','')}")
    print(f"  gold_count={result.gold_count}, pred_count={result.predicted_count}, "
          f"tp={result.true_positives}, f1={result.f1:.3f} (was {old_f1:.3f})")
    print(f"  mapping: {result.variable_mapping}")
    if result.aggregate_details:
        print(f"  aggregate: {result.aggregate_details}")

    # Execute both and show sample results
    g = strip_limit_offset(gold_sparql) if strip_limit else gold_sparql
    p = strip_limit_offset(pred_sparql) if strip_limit else pred_sparql

    g_exec = execute_query_full(g, endpoint, 30, 5)
    p_exec = execute_query_full(p, endpoint, 30, 5)

    print(f"\n  GOLD ({result.gold_count} rows total, showing up to 5):")
    if g_exec.success and g_exec.results:
        for row in g_exec.results[:5]:
            print(f"    {dict(row)}")
    elif not g_exec.success:
        print(f"    ERROR: {g_exec.error}")
    else:
        print("    (empty)")

    print(f"\n  PREDICTED ({result.predicted_count} rows total, showing up to 5):")
    if p_exec.success and p_exec.results:
        for row in p_exec.results[:5]:
            print(f"    {dict(row)}")
    elif not p_exec.success:
        print(f"    ERROR: {p_exec.error}")
    else:
        print("    (empty)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Replay F1 evaluation from stored report")
    parser.add_argument("report", type=Path, help="Existing F1 report JSON")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).parent.parent / "data" / "test_dataset.json",
        help="Test dataset (default: data/test_dataset.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: reports/<original>_replayed.json)",
    )
    parser.add_argument(
        "--no-strip-limit", action="store_true",
        help="Do not strip LIMIT/OFFSET from queries",
    )
    parser.add_argument(
        "--show-failures", action="store_true",
        help="Print gold vs predicted comparison for F1=0 cases",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print gold vs predicted comparison for ALL cases",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="SPARQL query timeout in seconds",
    )
    args = parser.parse_args()

    report_path = args.report
    if not report_path.exists():
        print(f"Error: report not found: {report_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        stem = report_path.stem
        output_path = report_path.parent / f"{stem}_replayed.json"

    replay(
        report_path=report_path,
        dataset_path=args.dataset,
        output_path=output_path,
        strip_limit=not args.no_strip_limit,
        show_failures=args.show_failures,
        show_all=args.show_all,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
