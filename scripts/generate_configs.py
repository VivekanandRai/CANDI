#!/usr/bin/env python3
"""Generate experiment configs for the full model x dataset x attack matrix.

The repository ships a representative subset of configs. This script regenerates
either track's full matrix so you do not have to hand-edit JSON.

White-box (local Hugging Face models), via the orchestrator's config templates:

    python scripts/generate_configs.py white-box \\
        --models llama3.2-1b llama3.2-3b llama3.1-8b qwen3-4b gemma3-1b \\
        --datasets failsafeqa anah faitheval \\
        --attacks vanilla_gcg_synonym vanilla_autodan_synonym sra_suffix_3 vanilla_pair vanilla_seca

Black-box (API models via OpenRouter), by cloning a shipped config and swapping
the target model and dataset:

    python scripts/generate_configs.py black-box \\
        --models openai/gpt-5-nano openai/gpt-5-mini openai/gpt-5-chat \\
                 google/gemini-2.5-flash-lite minimax/minimax-m2.1 \\
        --datasets anah failsafeqa faitheval \\
        --methods pair seca

Use --dry-run to list what would be written without touching the filesystem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WHITE_BOX_DIR = REPO_ROOT / "white_box_model_experiments"
BLACK_BOX_CONFIG_DIR = REPO_ROOT / "black_box_model_experiments" / "experiment_configs"

DATASETS = ("failsafeqa", "anah", "faitheval")

# Black-box configs differ only in the target model and the dataset, so each
# method's matrix is generated from one shipped config used as a template.
BLACK_BOX_TEMPLATES = {
    "pair": "pair_openai_gpt-5-nano_anah.json",
    "seca": "seca_openai_gpt-5-nano_anah.json",
}


def _slug(model: str) -> str:
    """openai/gpt-5-nano -> openai_gpt-5-nano (matches the shipped filenames)."""
    return model.replace("/", "_")


def generate_black_box(models, datasets, methods, output_dir: Path, dry_run: bool) -> int:
    written = 0
    for method in methods:
        template_path = BLACK_BOX_CONFIG_DIR / BLACK_BOX_TEMPLATES[method]
        if not template_path.exists():
            raise FileNotFoundError(f"Template config not found: {template_path}")
        template = json.loads(template_path.read_text())

        for model in models:
            for dataset in datasets:
                config = dict(template)
                config["model_name"] = model
                config["dataset"] = dataset

                name = f"{method}_{_slug(model)}_{dataset}.json"
                destination = output_dir / name
                if dry_run:
                    print(f"[dry-run] would write {destination.relative_to(REPO_ROOT)}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(config, indent=2) + "\n")
                    print(f"wrote {destination.relative_to(REPO_ROOT)}")
                written += 1
    return written


def generate_white_box(models, datasets, attacks, output_dir: Path, num_queries, dry_run: bool) -> int:
    # The orchestrator owns the per-attack hyperparameter defaults; reuse them
    # rather than duplicating a second source of truth here.
    sys.path.insert(0, str(WHITE_BOX_DIR))
    from orchestrator.config_templates import ATTACK_TYPES, generate_config

    unknown = [a for a in attacks if a not in ATTACK_TYPES]
    if unknown:
        raise ValueError(f"Unknown attack type(s): {unknown}. Available: {sorted(ATTACK_TYPES)}")

    written = 0
    for model in models:
        for dataset in datasets:
            for attack in attacks:
                config = generate_config(
                    model_name=model,
                    dataset=dataset,
                    attack_type=attack,
                    num_queries=num_queries,
                )
                name = f"{model}-{attack.replace('_', '-')}-{dataset}.json"
                destination = output_dir / name
                if dry_run:
                    print(f"[dry-run] would write {destination.relative_to(REPO_ROOT)}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(config, indent=4) + "\n")
                    print(f"wrote {destination.relative_to(REPO_ROOT)}")
                written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="track", required=True)
    dry_run_help = "List what would be written and exit without touching the filesystem."

    wb = sub.add_parser("white-box", help="Generate configs for local Hugging Face models.")
    wb.add_argument("--models", nargs="+", required=True, help="Short model names, e.g. llama3.2-1b.")
    wb.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    wb.add_argument("--attacks", nargs="+", required=True, help="Attack types from orchestrator ATTACK_TYPES.")
    wb.add_argument("--num-queries", type=int, default=30)
    wb.add_argument("--output-dir", type=Path, default=WHITE_BOX_DIR / "configs" / "generated")
    wb.add_argument("--dry-run", action="store_true", help=dry_run_help)

    bb = sub.add_parser("black-box", help="Generate configs for OpenRouter-hosted API models.")
    bb.add_argument("--models", nargs="+", required=True, help="OpenRouter model ids, e.g. openai/gpt-5-nano.")
    bb.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    bb.add_argument("--methods", nargs="+", default=["pair", "seca"], choices=["pair", "seca"])
    bb.add_argument("--output-dir", type=Path, default=BLACK_BOX_CONFIG_DIR)
    bb.add_argument("--dry-run", action="store_true", help=dry_run_help)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.track == "black-box":
        count = generate_black_box(args.models, args.datasets, args.methods, args.output_dir, args.dry_run)
    else:
        count = generate_white_box(
            args.models, args.datasets, args.attacks, args.output_dir, args.num_queries, args.dry_run
        )

    print(f"\n{'Would generate' if args.dry_run else 'Generated'} {count} config(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
