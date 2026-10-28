#!/usr/bin/env python3
"""
Summarize black-box experiment results (SECA and/or PAIR) as tables.

For every (method, model, dataset) the newest *complete* run (one with a summary.json)
is used. Metrics are recomputed from the per-query files with the runners' own
compute_summary, so every run is measured the same way regardless of which code
version wrote its summary.

Tables printed:
  1. Runs used, with failure counts
  2. Per model x dataset x category results
  3. Per model x category (pooled over datasets)
  4. Per category (pooled over models and datasets)

Usage:
  uv run python scripts/summarize_results.py                  # SECA and PAIR
  uv run python scripts/summarize_results.py --method seca
  uv run python scripts/summarize_results.py --csv results.csv
  uv run python scripts/summarize_results.py --output results.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BLACK_BOX_DIR = REPO_ROOT / "black_box_model_experiments"
sys.path.insert(0, str(BLACK_BOX_DIR))

from run_pair import compute_summary as compute_pair_summary  # noqa: E402
from run_seca import compute_summary as compute_seca_summary  # noqa: E402

CATEGORIES = ["active_passive", "hypernym", "synonym", "clause"]
COMPUTE_SUMMARY = {"seca": compute_seca_summary, "pair": compute_pair_summary}
RUN_DIR_RE = re.compile(r"^vanilla_(?P<method>[a-z]+)__(?P<model>.+)__(?P<dataset>[a-z]+)__(?P<ts>\d{8}_\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SECA/PAIR results as tables.")
    parser.add_argument("--method", choices=["seca", "pair", "all"], default="all")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=BLACK_BOX_DIR / "experiments",
        help="Directory containing the seca/ and pair/ run folders.",
    )
    parser.add_argument("--csv", type=Path, help="Also write the per model x dataset x category table as CSV.")
    parser.add_argument("--output", type=Path, help="Also write the markdown tables to this file.")
    return parser.parse_args()


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for q_file in sorted(run_dir.glob("query_*.json")):
        try:
            with open(q_file) as f:
                row = json.load(f)
        except Exception as exc:
            print(f"warning: could not read {q_file}: {exc}", file=sys.stderr)
            continue
        if "transformation_applied" not in row:
            # Rows written before this field existed.
            adversarial = row.get("adversarial_query")
            row["transformation_applied"] = bool(adversarial) and (
                str(adversarial).strip() != str(row.get("original_query", "")).strip()
            )
        rows.append(row)
    return rows


def find_runs(experiments_dir: Path, method: str) -> Dict[Tuple[str, str], Path]:
    """Return the newest complete run directory per (model, dataset)."""
    method_dir = experiments_dir / method
    latest: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    if not method_dir.is_dir():
        return {}
    for run_dir in method_dir.iterdir():
        match = RUN_DIR_RE.match(run_dir.name)
        if not match or not (run_dir / "summary.json").exists():
            continue
        key = (match["model"], match["dataset"])
        if key not in latest or match["ts"] > latest[key][0]:
            latest[key] = (match["ts"], run_dir)
    return {key: run_dir for key, (_ts, run_dir) in latest.items()}


def pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def frac(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return f"{100 * numerator / denominator:.1f}% ({numerator}/{denominator})"


def num(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def category_cells(stats: Dict[str, Any]) -> Dict[str, str]:
    ok = stats["successful_runs"]
    initially_clean = ok - stats["initial_hallucination_count"]
    return {
        "OK/Failed": f"{ok}/{stats['failed_runs']}",
        "Initial halluc.": frac(stats["initial_hallucination_count"], ok),
        "Final halluc.": frac(stats["final_hallucination_count"], ok),
        "ASR": frac(stats["attack_success_count"], initially_clean),
        "Transformed": frac(stats["transformation_applied_count"], ok),
        "Mean score": num(stats["mean_best_hallucination_score"]),
    }


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


METRIC_HEADERS = ["OK/Failed", "Initial halluc.", "Final halluc.", "ASR", "Transformed", "Mean score"]


def summarize_method(method: str, experiments_dir: Path, csv_rows: List[Dict[str, Any]]) -> str:
    runs = find_runs(experiments_dir, method)
    if not runs:
        return f"## {method.upper()}\n\nNo complete runs found in {experiments_dir / method}.\n"

    compute_summary = COMPUTE_SUMMARY[method]
    rows_by_key = {key: load_rows(run_dir) for key, run_dir in sorted(runs.items())}
    sections = [f"## {method.upper()}"]

    # 1. Runs used
    run_rows = []
    legacy_runs = False
    for (model, dataset), run_dir in sorted(runs.items()):
        rows = rows_by_key[(model, dataset)]
        failed = [r for r in rows if r.get("error") not in (None, "")]
        first_error = str(failed[0]["error"])[:90] if failed else ""
        legacy = any("category" not in r for r in rows) or not rows
        try:
            with open(run_dir / "query_00000_active_passive.json") as f:
                legacy = legacy or "transformation_applied" not in json.load(f)
        except Exception:
            pass
        legacy_runs = legacy_runs or legacy
        run_rows.append([
            model, dataset, run_dir.name.rsplit("__", 1)[-1] + (" *" if legacy else ""),
            str(len(rows)), str(len(failed)), first_error,
        ])
    sections.append("### Runs used\n\n" + markdown_table(
        ["Model", "Dataset", "Run", "Rows", "Failed", "First error"], run_rows
    ))
    if legacy_runs:
        sections.append(
            "\\* Written by code without the category-compliance check; "
            "its per-category numbers are not directly comparable."
        )

    # 2. Model x dataset x category
    detail_rows = []
    for (model, dataset), rows in rows_by_key.items():
        summary = compute_summary(rows)
        for category in CATEGORIES:
            stats = summary["by_category"][category]
            cells = category_cells(stats)
            detail_rows.append([model, dataset, category] + [cells[h] for h in METRIC_HEADERS])
            csv_rows.append({
                "method": method, "model": model, "dataset": dataset, "category": category,
                "run_dir": runs[(model, dataset)].name,
                **{k: v for k, v in stats.items() if k != "errors"},
            })
    sections.append("### Results by model, dataset and category\n\n" + markdown_table(
        ["Model", "Dataset", "Category"] + METRIC_HEADERS, detail_rows
    ))

    # 3. Model x category, pooled over datasets
    models = sorted({model for model, _ in rows_by_key})
    model_rows = []
    for model in models:
        pooled = [r for (m, _d), rows in rows_by_key.items() if m == model for r in rows]
        summary = compute_summary(pooled)
        for category in CATEGORIES:
            cells = category_cells(summary["by_category"][category])
            model_rows.append([model, category] + [cells[h] for h in METRIC_HEADERS])
    sections.append("### Results by model and category (all datasets)\n\n" + markdown_table(
        ["Model", "Category"] + METRIC_HEADERS, model_rows
    ))

    # 4. Category, pooled over everything
    all_rows = [r for rows in rows_by_key.values() for r in rows]
    summary = compute_summary(all_rows)
    category_rows = []
    for category in CATEGORIES:
        cells = category_cells(summary["by_category"][category])
        category_rows.append([category] + [cells[h] for h in METRIC_HEADERS])
    sections.append("### Results by category (all models and datasets)\n\n" + markdown_table(
        ["Category"] + METRIC_HEADERS, category_rows
    ))

    return "\n\n".join(sections) + "\n"


LEGEND = """\
Metrics are computed over runs without errors (failed runs are excluded).
- OK/Failed: runs that completed / runs that errored
- Initial halluc.: original query produced a hallucination
- Final halluc.: final (best) query produced a hallucination
- ASR: share of initially non-hallucinating queries that hallucinated after the attack
- Transformed: final query differs from the original (a compliant rewrite was found)
- Mean score: mean best hallucination score (0-10)
"""


def main() -> None:
    args = parse_args()
    methods = ["seca", "pair"] if args.method == "all" else [args.method]
    csv_rows: List[Dict[str, Any]] = []

    report = "# Linguistic transformation results\n\n" + LEGEND + "\n"
    report += "\n".join(summarize_method(m, args.experiments_dir, csv_rows) for m in methods)
    print(report)

    if args.output:
        args.output.write_text(report)
        print(f"Wrote {args.output}", file=sys.stderr)
    if args.csv and csv_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
