#!/usr/bin/env python3
"""
Results Aggregator - Extract and aggregate experiment results

Scans experiment directories for overall_results.json files and generates
comprehensive CSV reports with all metrics.

Usage:
    # Aggregate single experiment (auto-saves to experiment_dir/aggregated_results.csv)
    python orchestrator/results_aggregator.py \
        --experiment_dir experiments/exp_001_baseline
    
    # Aggregate with custom output file
    python orchestrator/results_aggregator.py \
        --experiment_dir experiments/exp_001_baseline \
        --output_file results.csv
    
    # Aggregate multiple experiments
    python orchestrator/results_aggregator.py \
        --experiment_dirs experiments/exp_001_* experiments/exp_002_* \
        --output_file combined_results.csv
    
    # With specific format
    python orchestrator/results_aggregator.py \
        --experiment_dir experiments/exp_001_baseline \
        --format json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import glob

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.utils import load_json


class ResultsAggregator:
    """Aggregates experiment results into structured formats."""
    
    def __init__(
        self,
        experiment_dirs: List[Path],
        output_file: Path,
        output_format: str = "csv"
    ):
        """
        Initialize the aggregator.
        
        Args:
            experiment_dirs: List of experiment directory paths
            output_file: Path to output file
            output_format: Output format (csv, json, markdown)
        """
        self.experiment_dirs = experiment_dirs
        self.output_file = output_file
        self.output_format = output_format
        self.results = []
        
    def discover_results(self) -> List[Dict[str, Any]]:
        """
        Discover all overall_results.json files in experiment directories.
        
        Returns:
            List of dicts with result file paths and metadata
        """
        discovered = []
        
        for exp_dir in self.experiment_dirs:
            if not exp_dir.exists():
                print(f"Warning: Directory not found: {exp_dir}")
                continue
            
            # Try to load manifest
            manifest_path = exp_dir / "job_manifest.json"
            manifest = None
            if manifest_path.exists():
                try:
                    manifest = load_json(manifest_path)
                except Exception as e:
                    print(f"Warning: Could not load manifest from {manifest_path}: {e}")
            
            # Find all overall_results.json files
            # Check both exp_dir/logs/ (new structure) and exp_dir/ (old structure)
            search_dirs = []
            logs_dir = exp_dir / "logs"
            if logs_dir.exists():
                search_dirs.append(logs_dir)
            else:
                # Old structure: logs directly in exp_dir
                search_dirs.append(exp_dir)
            
            for search_dir in search_dirs:
                for results_file in search_dir.rglob("overall_results.json"):
                    discovered.append({
                        "experiment_dir": exp_dir,
                        "experiment_name": exp_dir.name,
                        "results_file": results_file,
                        "manifest": manifest
                    })
        
        return discovered
    
    def extract_metrics(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metrics from overall_results.json.
        
        Args:
            results_data: Loaded JSON data
            
        Returns:
            Dictionary with extracted metrics
        """
        summary = results_data.get("summary", {})
        config = results_data.get("config", {})
        individual_results = results_data.get("results", [])
        
        # Extract basic info
        metrics = {
            "model_name": summary.get("model_name", ""),
            "hf_model_name": summary.get("hf_model_name", ""),
            "base_method": summary.get("base_method", ""),
            "method": summary.get("method", ""),
            "config_path": summary.get("config_path", ""),
            "dataset_path": summary.get("dataset_path", ""),
        }
        
        # Extract hyperparameters from config
        base_method = config.get("base_method", "")
        if base_method == "gcg" and "gcg" in config:
            gcg_common = config["gcg"].get("common", {})
            metrics.update({
                "num_steps": gcg_common.get("num_steps", ""),
                "search_width": gcg_common.get("search_width", ""),
                "batch_size": gcg_common.get("batch_size", ""),
                "optim_str_mode": gcg_common.get("optim_str_mode", ""),
                "optim_str_length": gcg_common.get("optim_str_length", ""),
                "optim_str_constraint": gcg_common.get("optim_str_constraint", ""),
                "stagnation_patience": gcg_common.get("stagnation_patience", ""),
                "top_k": gcg_common.get("top_k", ""),
            })
        elif base_method == "autodan" and "autodan" in config:
            autodan_common = config["autodan"].get("common", {})
            metrics.update({
                "num_steps": autodan_common.get("num_steps", ""),
                "batch_size": autodan_common.get("batch_size", ""),
                "stagnation_patience": autodan_common.get("stagnation_patience", ""),
                "num_elites": autodan_common.get("num_elites", ""),
                "mutation_rate": autodan_common.get("mutation_rate", ""),
                "optim_str_mode": autodan_common.get("optim_str_mode", ""),
                "optim_str_constraint": autodan_common.get("optim_str_constraint", ""),
                "stagnation_patience": autodan_common.get("stagnation_patience", ""),
                
            })
        elif base_method == "sra" and "sra" in config:
            sra_common = config["sra"].get("common", {})
            metrics.update({
                "batch_size": sra_common.get("batch_size", ""),
                "stagnation_patience": sra_common.get("stagnation_patience", ""),
                "prompt_length_threshold": sra_common.get("prompt_length_threshold", ""),
                "top_p": sra_common.get("top_p", ""),
                "top_k": sra_common.get("top_k", ""),
                "optim_str_mode": "suffix",
                "optim_str_length": sra_common.get("prompt_length_threshold", ""),
            })
        elif base_method == "pair" and "pair" in config:
            pair_common = config["pair"].get("common", {})
            metrics.update({
                "n_streams": pair_common.get("n_streams", ""),
                "n_iterations": pair_common.get("n_iterations", ""),
                "semantic_similarity_threshold": pair_common.get("semantic_similarity_threshold", ""),
            })
        elif base_method == "seca" and "seca" in config:
            seca_common = config["seca"].get("common", {})
            metrics.update({
                "max_iteration": seca_common.get("max_iteration", ""),
                "top_N_most_adversarial": seca_common.get("top_N_most_adversarial", ""),
                "candidate_size_M": seca_common.get("candidate_size_M", ""),
                "stagnation_patience": seca_common.get("stagnation_patience", ""),
            })
        
        # Extract summary metrics
        metrics.update({
            "total_queries": summary.get("total_queries", ""),
            "initial_correct": summary.get("initial_correct", ""),
            "final_correct": summary.get("final_correct", ""),
            "attack_success_count": summary.get("attack_success_count", ""),
            "initial_accuracy": summary.get("initial_accuracy", ""),
            "final_accuracy": summary.get("final_accuracy", ""),
            "attack_success_rate": summary.get("attack_success_rate", ""),
            "avg_time_per_query_seconds": summary.get("avg_time_per_query_seconds", ""),
            "total_time_seconds": summary.get("total_time_seconds", ""),
        })
        
        # Extract similarity and equivalence metrics from summary
        metrics.update({
            "mean_cosine_similarity": summary.get("mean_cosine_similarity", ""),
            "mean_semantic_equivalence": summary.get("mean_semantic_equivalence_is_equivalent", ""),
            "mean_adversarial_query_perplexity": summary.get("mean_adversarial_query_perplexity", ""),
            "mean_rouge1_f1": summary.get("mean_rouge1_f1", ""),
            "mean_rouge2_f1": summary.get("mean_rouge2_f1", ""),
            "mean_rougeL_f1": summary.get("mean_rougeL_f1", ""),
        })
        
        # Compute equivalence metrics from individual results (for backwards compatibility)
        if individual_results:
            metrics.update(self._compute_equivalence_metrics(individual_results))
        
        return metrics
    
    def _compute_equivalence_metrics(
        self,
        individual_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute average equivalence metrics from individual results.
        
        Args:
            individual_results: List of individual query results
            
        Returns:
            Dictionary with averaged metrics
        """
        metrics = {}
        
        # Metrics to compute averages for
        metric_keys = [
            "perplexity",
            "edit_distance",
            "semantic_similarity",
            "bleu_score",
            "token_overlap",
            "character_overlap",
            "word_overlap"
        ]
        
        for key in metric_keys:
            values = []
            for result in individual_results:
                if key in result and result[key] is not None:
                    try:
                        values.append(float(result[key]))
                    except (ValueError, TypeError):
                        pass
            
            if values:
                avg_key = f"avg_{key}"
                metrics[avg_key] = sum(values) / len(values)
            else:
                metrics[f"avg_{key}"] = ""
        
        # Compute average number of steps
        steps = []
        for result in individual_results:
            if "steps" in result and result["steps"] is not None:
                try:
                    steps.append(int(result["steps"]))
                except (ValueError, TypeError):
                    pass
        
        if steps:
            metrics["avg_steps"] = sum(steps) / len(steps)
        else:
            metrics["avg_steps"] = ""
        
        return metrics
    
    def process_results(self, discovered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all discovered results and extract metrics.
        
        Args:
            discovered: List of discovered result files
            
        Returns:
            List of processed result dictionaries
        """
        processed = []
        
        for item in discovered:
            try:
                results_data = load_json(item["results_file"])
                metrics = self.extract_metrics(results_data)
                
                # Add experiment metadata
                metrics["experiment_name"] = item["experiment_name"]
                metrics["results_file"] = str(item["results_file"].relative_to(item["experiment_dir"]))
                
                # Determine attack type from method and constraint
                attack_type = self._infer_attack_type(results_data.get("config", {}))
                metrics["attack_type"] = attack_type
                
                # Add status
                metrics["status"] = "completed"
                
                processed.append(metrics)
                
                print(f"✓ Processed: {item['results_file'].name}")
                
            except Exception as e:
                print(f"✗ Error processing {item['results_file']}: {e}")
                continue
        
        return processed
    
    def _infer_attack_type(self, config: Dict[str, Any]) -> str:
        """
        Infer attack type from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Attack type string
        """
        method = config.get("method", "")
        base_method = config.get("base_method", "")
        
        # For GCG, check constraint
        if base_method == "gcg" and "gcg" in config:
            gcg_common = config["gcg"].get("common", {})
            optim_mode = gcg_common.get("optim_str_mode", "")
            constraint = gcg_common.get("optim_str_constraint", "")
            
            if optim_mode == "suffix":
                suffix_type = "suffix"
            elif constraint == "synonym_replacement":
                suffix_type = "synonym"
            elif constraint == "punctuation_only":
                suffix_type = "punctuation"
            else:
                suffix_type = "suffix"
            
            return f"{method}_{suffix_type}"
        
        # For SRA, check prompt_length_threshold
        elif base_method == "sra" and "sra" in config:
            sra_common = config["sra"].get("common", {})
            threshold = sra_common.get("prompt_length_threshold", 3)
            return f"sra_suffix_{threshold}"
        
        # For other methods, just return method name
        else:
            return method
    
    def write_csv(self, results: List[Dict[str, Any]]) -> None:
        """
        Write results to CSV file.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            print("No results to write")
            return
        
        # Define column order
        columns = [
            "experiment_name",
            "model_name",
            "hf_model_name",
            "base_method",
            "method",
            "attack_type",
            "config_path",
            "dataset_path",
            "num_steps",
            "search_width",
            "batch_size",
            "optim_str_mode",
            "optim_str_length",
            "optim_str_constraint",
            "stagnation_patience",
            "top_k",
            "total_queries",
            "initial_correct",
            "final_correct",
            "attack_success_count",
            "initial_accuracy",
            "final_accuracy",
            "attack_success_rate",
            "avg_steps",
            "mean_adversarial_query_perplexity",
            "mean_cosine_similarity",
            "mean_semantic_equivalence",
            "mean_rouge1_f1",
            "mean_rouge2_f1",
            "mean_rougeL_f1",
            "avg_perplexity",
            "avg_edit_distance",
            "avg_semantic_similarity",
            "avg_bleu_score",
            "avg_token_overlap",
            "avg_character_overlap",
            "avg_word_overlap",
            "avg_time_per_query_seconds",
            "total_time_seconds",
            "results_file",
            "status"
        ]
        
        # Get all unique keys from results (in case some have extra fields)
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Add any extra columns not in our predefined list
        extra_columns = sorted(list(all_keys - set(columns)))
        columns.extend(extra_columns)
        
        # Write CSV
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Wrote {len(results)} results to {self.output_file}")
    
    def write_json(self, results: List[Dict[str, Any]]) -> None:
        """
        Write results to JSON file.
        
        Args:
            results: List of result dictionaries
        """
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Wrote {len(results)} results to {self.output_file}")
    
    def write_markdown(self, results: List[Dict[str, Any]]) -> None:
        """
        Write results to Markdown table.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            print("No results to write")
            return
        
        # Select key columns for markdown
        columns = [
            "experiment_name",
            "model_name",
            "method",
            "attack_type",
            "total_queries",
            "initial_accuracy",
            "final_accuracy",
            "attack_success_rate",
            "avg_time_per_query_seconds"
        ]
        
        lines = []
        
        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.extend([header, separator])
        
        # Rows
        for result in results:
            row_values = []
            for col in columns:
                value = result.get(col, "")
                # Format floats
                if isinstance(value, float):
                    value = f"{value:.4f}"
                row_values.append(str(value))
            
            row = "| " + " | ".join(row_values) + " |"
            lines.append(row)
        
        # Write to file
        with open(self.output_file, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"\n✓ Wrote {len(results)} results to {self.output_file}")
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """
        Print summary statistics.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            return
        
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}\n")
        
        print(f"Total experiments: {len(results)}")
        
        # Group by experiment
        experiments = set(r["experiment_name"] for r in results)
        print(f"Unique experiment names: {len(experiments)}")
        
        # Group by model
        models = set(r.get("model_name", "") for r in results if r.get("model_name"))
        print(f"Models tested: {', '.join(sorted(models))}")
        
        # Group by method
        methods = set(r.get("method", "") for r in results if r.get("method"))
        print(f"Attack methods: {', '.join(sorted(methods))}")
        
        # Compute statistics
        if results:
            asr_values = [r.get("attack_success_rate", 0) for r in results if r.get("attack_success_rate") != ""]
            if asr_values:
                avg_asr = sum(asr_values) / len(asr_values)
                print(f"\nAverage attack success rate: {avg_asr:.2%}")
                print(f"Min ASR: {min(asr_values):.2%}")
                print(f"Max ASR: {max(asr_values):.2%}")
    
    def run(self) -> None:
        """Execute the aggregation workflow."""
        print(f"\n{'='*60}")
        print("Results Aggregator")
        print(f"{'='*60}\n")
        
        print(f"Scanning {len(self.experiment_dirs)} experiment director{'y' if len(self.experiment_dirs) == 1 else 'ies'}...")
        
        # Discover results
        discovered = self.discover_results()
        print(f"Found {len(discovered)} result files\n")
        
        if not discovered:
            print("No results found. Exiting.")
            return
        
        # Process results
        print("Processing results...")
        results = self.process_results(discovered)
        
        if not results:
            print("No valid results to aggregate. Exiting.")
            return
        
        # Write output
        print(f"\nWriting results to {self.output_file}...")
        if self.output_format == "csv":
            self.write_csv(results)
        elif self.output_format == "json":
            self.write_json(results)
        elif self.output_format == "markdown":
            self.write_markdown(results)
        
        # Print summary
        self.print_summary(results)


def expand_glob_patterns(patterns: List[str]) -> List[Path]:
    """
    Expand glob patterns to actual paths.
    
    Args:
        patterns: List of path patterns (may include globs)
        
    Returns:
        List of Path objects
    """
    paths = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            paths.extend([Path(p) for p in matched])
        else:
            # Not a glob, treat as literal path
            paths.append(Path(pattern))
    
    return paths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Results Aggregator - Extract and aggregate experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate single experiment (auto-saves to experiment_dir/aggregated_results.csv)
  python orchestrator/results_aggregator.py \\
      --experiment_dir experiments/exp_001_baseline
  
  # Aggregate with custom output file
  python orchestrator/results_aggregator.py \\
      --experiment_dir experiments/exp_001_baseline \\
      --output_file results.csv
  
  # Aggregate multiple experiments
  python orchestrator/results_aggregator.py \\
      --experiment_dirs experiments/exp_001_* experiments/exp_002_* \\
      --output_file combined_results.csv
  
  # Output as JSON (auto-saves to experiment_dir/aggregated_results.json)
  python orchestrator/results_aggregator.py \\
      --experiment_dir experiments/exp_001_baseline \\
      --format json
  
  # Output as Markdown table with custom file
  python orchestrator/results_aggregator.py \\
      --experiment_dir experiments/exp_001_baseline \\
      --output_file results.md \\
      --format markdown
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--experiment_dir",
        type=str,
        help="Single experiment directory to process"
    )
    input_group.add_argument(
        "--experiment_dirs",
        type=str,
        nargs="+",
        help="Multiple experiment directories (supports glob patterns)"
    )
    
    # Output options
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Output file path (default: auto-generated in experiment directory)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "markdown"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    args = parser.parse_args()
    
    # Determine experiment directories
    if args.experiment_dir:
        experiment_dirs = [Path(args.experiment_dir)]
    else:
        experiment_dirs = expand_glob_patterns(args.experiment_dirs)
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # Auto-generate output file in the first experiment directory
        base_dir = experiment_dirs[0]
        ext_map = {"csv": ".csv", "json": ".json", "markdown": ".md"}
        ext = ext_map.get(args.format, ".csv")
        output_file = base_dir / f"aggregated_results{ext}"
        print(f"No output file specified. Auto-saving to: {output_file}")
    
    # Create aggregator and run
    aggregator = ResultsAggregator(
        experiment_dirs=experiment_dirs,
        output_file=output_file,
        output_format=args.format
    )
    
    try:
        aggregator.run()
    except Exception as e:
        print(f"\n❌ Error during aggregation:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
