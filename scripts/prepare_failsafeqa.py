#!/usr/bin/env python3
"""Prepare the FailSafeQA benchmark slice shipped with this repository.

The upstream FailSafeQA release contains 220 long-context SEC-filing records and
is ~54 MiB, which is awkward to keep in git. The experiments in the paper use the
first 100 records, so that is what the repository ships.

This script regenerates ``dataset/failsafeqa_benchmark_data.json`` from a full
export. It is only needed if you want to rebuild or re-slice the file; the
prepared dataset is already committed.

Usage:
    python scripts/prepare_failsafeqa.py --source /path/to/full_failsafeqa.json
    python scripts/prepare_failsafeqa.py --source ... --limit 220        # full set
    python scripts/prepare_failsafeqa.py --source ... --slim             # ~13 MB

The ``--slim`` flag drops ``ocr_context``, which no code path in this repository
reads (the attacks use ``query``, ``answer`` and ``citations``). It roughly halves
the file size at the cost of no longer being a faithful copy of upstream records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "failsafeqa_benchmark_data.json"

# Fields the attack code actually consumes. Everything else is upstream metadata.
USED_FIELDS = ("idx", "query", "answer", "citations", "context")
SLIM_DROP_FIELDS = ("ocr_context",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the full FailSafeQA export (a JSON array of records).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the prepared dataset (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of leading records to keep (default: 100, matching the shipped slice).",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help=f"Drop unused fields {SLIM_DROP_FIELDS} to roughly halve the file size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source}")

    # Note: upstream ships this as a ".jsonl" file, but the contents are a
    # standard JSON array, not one object per line.
    with args.source.open() as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array of records, got {type(records).__name__}")

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    subset = records[: args.limit]
    if len(subset) < args.limit:
        print(f"WARNING: source has only {len(records)} records; keeping all of them.")

    if args.slim:
        subset = [{k: v for k, v in r.items() if k not in SLIM_DROP_FIELDS} for r in subset]

    missing = [f for f in ("query", "answer", "citations") if f not in subset[0]]
    if missing:
        raise ValueError(f"Records are missing required field(s): {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(subset, f)

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {len(subset)} records to {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
