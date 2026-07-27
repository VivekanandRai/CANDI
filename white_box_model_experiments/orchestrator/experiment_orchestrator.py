#!/usr/bin/env python3
"""
Experiment Orchestrator - Main Script

Generates experiment configurations and either PBS job scripts or a single
sequential bash script from high-level parameters.  Multiple datasets can
be swept in a single invocation.

Usage:
    python orchestrator/experiment_orchestrator.py \
        --experiment_name "baseline_attacks" \
        --model_names "llama3.2-1b" "llama3-8b" \
        --dataset "failsafeqa" "anah" \
        --attack_types "vanilla_gcg_suffix" "ecs_gcg_suffix" \
        --num_queries 30 \
        --output_base_dir "experiments" \
        --script_mode bash
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.config_templates import (
    generate_config,
    ATTACK_TYPES,
    MODEL_NAMES,
    DATASET_NAMES,
    get_attack_description
)
from orchestrator.resource_mappings import (
    get_job_resources,
    format_pbs_resources,
    format_walltime,
    MODEL_GPU_REQUIREMENTS
)
from orchestrator.utils import (
    DEFAULT_CONDA_ENV,
    DEFAULT_CONDA_HOOK,
    create_experiment_directory,
    get_config_name,
    get_job_name,
    save_json,
    ExperimentManifest,
    format_timestamp
)


class ExperimentOrchestrator:
    """Main orchestrator class for experiment generation."""
    
    def __init__(
        self,
        experiment_name: str,
        model_names: List[str],
        datasets: Union[str, List[str]],
        attack_types: List[str],
        output_base_dir: str = "experiments",
        num_queries: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
        conda_env: str = None,
        dry_run: bool = False,
        script_mode: str = "pbs"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            experiment_name: Name for this experiment batch
            model_names: List of model names to test
            datasets: Dataset name or list of dataset names to sweep
            attack_types: List of attack types to run
            output_base_dir: Base directory for experiments
            num_queries: Number of queries (None = use defaults)
            overrides: Dictionary of config overrides
            conda_env: Conda environment name
            dry_run: If True, don't create files
            script_mode: "pbs" to generate individual PBS job scripts;
                         "bash" to generate one sequential bash script
        """
        self.experiment_name = experiment_name
        self.model_names = model_names
        self.datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        self.attack_types = attack_types
        self.output_base_dir = Path(output_base_dir)
        self.num_queries = num_queries
        self.overrides = overrides or {}
        self.conda_env = conda_env or DEFAULT_CONDA_ENV
        self.dry_run = dry_run
        self.script_mode = script_mode
        
        # Will be set during execution
        self.experiment_dir = None
        self.directories = {}
        self.manifest = None
        self.jobs = []
        
        # Get project root (where run.py is located)
        self.project_root = Path(__file__).resolve().parent.parent
        
    def validate_inputs(self) -> None:
        """Validate all input parameters."""
        errors = []
        
        # Validate models
        for model in self.model_names:
            if model not in MODEL_NAMES:
                errors.append(
                    f"Unknown model: {model}. "
                    f"Available: {list(MODEL_NAMES.keys())}"
                )
        
        # Validate datasets
        for dataset in self.datasets:
            if dataset not in DATASET_NAMES:
                errors.append(
                    f"Unknown dataset: {dataset}. "
                    f"Available: {list(DATASET_NAMES.keys())}"
                )
        
        # Validate script_mode
        if self.script_mode not in ("pbs", "bash"):
            errors.append(
                f"Unknown script_mode: '{self.script_mode}'. Must be 'pbs' or 'bash'."
            )
        
        # Validate attack types
        for attack in self.attack_types:
            if attack not in ATTACK_TYPES:
                errors.append(
                    f"Unknown attack type: {attack}. "
                    f"Available: {list(ATTACK_TYPES.keys())}"
                )
        
        if errors:
            raise ValueError("\n".join(errors))
    
    def setup_directories(self) -> None:
        """Create experiment directory structure."""
        print(f"\n{'='*60}")
        print(f"Creating Experiment: {self.experiment_name}")
        print(f"{'='*60}\n")
        
        if not self.dry_run:
            self.experiment_dir = create_experiment_directory(
                self.output_base_dir,
                self.experiment_name
            )
            self.directories = {
                "base": self.experiment_dir,
                "configs": self.experiment_dir / "configs",
                "scripts": self.experiment_dir / "scripts",
                "logs": self.experiment_dir / "logs"
            }
            
            print(f"✓ Created experiment directory: {self.experiment_dir}")
        else:
            exp_num = 1  # Placeholder for dry run
            dir_name = f"exp_{exp_num:03d}_{self.experiment_name}"
            self.experiment_dir = self.output_base_dir / dir_name
            self.directories = {
                "base": self.experiment_dir,
                "configs": self.experiment_dir / "configs",
                "scripts": self.experiment_dir / "scripts",
                "logs": self.experiment_dir / "logs"
            }
            print(f"[DRY RUN] Would create: {self.experiment_dir}")
    
    def generate_configs(self) -> None:
        """Generate configuration files for all combinations."""
        print(f"\n{'='*60}")
        print(f"Generating Configuration Files")
        print(f"{'='*60}\n")
        
        config_count = 0
        multi_dataset = len(self.datasets) > 1
        
        for dataset in self.datasets:
            for model in self.model_names:
                for attack in self.attack_types:
                    # Generate config
                    config = generate_config(
                        model_name=model,
                        dataset=dataset,
                        attack_type=attack,
                        output_dir=None,  # Will be set below
                        num_queries=self.num_queries,
                        overrides=self.overrides
                    )
                    
                    # Include dataset in config name when sweeping multiple datasets
                    # to prevent filename collisions across datasets
                    base_name = get_config_name(model, attack)
                    config_name = f"{base_name}_{dataset}" if multi_dataset else base_name
                    
                    # Set output_dir to logs/config_name/ - run.py will append model_timestamp
                    # This ensures each job gets its own directory with a clear name
                    config["output_dir"] = str(self.directories["logs"] / config_name)
                    
                    # Generate filename
                    config_path = self.directories["configs"] / f"{config_name}.json"
                    
                    # Save config
                    if not self.dry_run:
                        save_json(config, config_path)
                        print(f"✓ Created config: {config_path.name}")
                    else:
                        print(f"[DRY RUN] Would create: {config_path.name}")
                    
                    # Store job info
                    job_info = {
                        "model": model,
                        "attack": attack,
                        "dataset": dataset,
                        "config_name": config_name,
                        "config_path": config_path
                    }
                    self.jobs.append(job_info)
                    config_count += 1
        
        print(f"\n✓ Generated {config_count} configuration files")
    
    def generate_pbs_scripts(self) -> None:
        """Generate PBS job scripts for all configs."""
        print(f"\n{'='*60}")
        print(f"Generating PBS Job Scripts")
        print(f"{'='*60}\n")
        
        script_count = 0
        
        for job in self.jobs:
            model = job["model"]
            attack = job["attack"]
            config_name = job["config_name"]
            config_path = job["config_path"]
            
            # Get resource requirements (scale walltime by num_queries)
            num_queries = self.num_queries if self.num_queries else 30
            resources = get_job_resources(model, attack, num_queries)
            
            # Generate PBS script
            script_content = self._generate_pbs_script(
                job_name=get_job_name(model, attack),
                config_path=config_path,
                resources=resources,
                config_name=config_name
            )
            
            # Save script
            script_path = self.directories["scripts"] / f"{config_name}.pbs"
            
            if not self.dry_run:
                script_path.write_text(script_content)
                # Make executable
                script_path.chmod(0o755)
                print(f"✓ Created script: {script_path.name}")
            else:
                print(f"[DRY RUN] Would create: {script_path.name}")
            
            # Add script path to job info
            job["script_path"] = script_path
            script_count += 1
        
        print(f"\n✓ Generated {script_count} PBS scripts")
    
    def _generate_pbs_script(
        self,
        job_name: str,
        config_path: Path,
        resources: Dict[str, int],
        config_name: str
    ) -> str:
        """
        Generate PBS script content.
        
        Args:
            job_name: PBS job name
            config_path: Path to config file
            resources: Resource requirements dict
            config_name: Config name for log file
            
        Returns:
            PBS script content as string
        """
        # Format resources
        pbs_resources = format_pbs_resources(resources)
        walltime = format_walltime(resources["walltime_hours"])
        
        # Get absolute paths
        config_abs_path = config_path.resolve()
        project_root = self.project_root
        
        # Create timestamp for log file
        timestamp_var = "$(date +\"%Y%m%d_%H%M%S\")"
        
        script = f"""#!/bin/bash
#PBS -l{pbs_resources}
#PBS -lwalltime={walltime}

# Conda environment activation
eval "$({DEFAULT_CONDA_HOOK} shell.bash hook)"
source activate {self.conda_env}

# Change to project directory
cd {project_root}

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
fi

# Verify critical environment variables are set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set"
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "WARNING: OPENROUTER_API_KEY is not set"
fi

# Create running marker file
MARKER_FILE="{self.experiment_dir.resolve()}/{config_name}_running.txt"
touch "$MARKER_FILE"
echo "Created marker file: $MARKER_FILE"

# Run experiment
TIMESTAMP={timestamp_var}

python run.py \
    --config {config_abs_path} \
    &> {self.experiment_dir.resolve()}/logs/{config_name}_${{TIMESTAMP}}.log 2>&1

# Remove running marker file
rm -f "$MARKER_FILE"
echo "Removed marker file: $MARKER_FILE"

# Print completion message
echo "Job completed at $(date)"
echo "Config: {config_abs_path}"
echo "Log: {self.experiment_dir.resolve()}/logs/{config_name}_${{TIMESTAMP}}.log"
"""
        return script
    
    def generate_bash_script(self) -> None:
        """Generate a single sequential bash script containing all runs."""
        print(f"\n{'='*60}")
        print(f"Generating Sequential Bash Script")
        print(f"{'='*60}\n")
        
        project_root = self.project_root
        exp_dir = self.experiment_dir.resolve()
        total = len(self.jobs)
        
        lines = [
            "#!/bin/bash",
            "# Auto-generated sequential experiment script",
            f"# Experiment: {self.experiment_name}",
            f"# Generated: {format_timestamp()}",
            f"# Total runs: {total}",
            "#",
            "# Runs all experiments sequentially.",
            "# Each run's stdout/stderr is captured to the logs/ directory.",
            "",
            "",
            f'cd {project_root}',
            "",
            'if [ -f .env ]; then',
            '    echo "Loading .env..."',
            '    set -a',
            '    source .env',
            '    set +a',
            'fi',
            "",
            "source .venv/bin/activate",
            "",
            'if [ -z "${HF_TOKEN:-}" ]; then',
            '    echo "WARNING: HF_TOKEN is not set"',
            'fi',
            'if [ -z "${OPENROUTER_API_KEY:-}" ]; then',
            '    echo "WARNING: OPENROUTER_API_KEY is not set"',
            'fi',
            "",
            f'echo "Starting {total} sequential runs for experiment: {self.experiment_name}"',
            'echo "' + "=" * 60 + '"',
            "",
        ]
        
        log_dir = exp_dir / "logs"
        for i, job in enumerate(self.jobs, 1):
            config_name = job["config_name"]
            config_abs = job["config_path"].resolve()
            lines += [
                'echo ""',
                f'echo "[{i}/{total}] {config_name}"',
                'TIMESTAMP=$(date +"%Y%m%d_%H%M%S")',
                f'LOG_FILE="{log_dir}/{config_name}_${{TIMESTAMP}}.log"',
                f'echo "  Config : {config_abs}"',
                'echo "  Log    : $LOG_FILE"',
                f'python run.py \\',
                f'    --config {config_abs} \\',
                '    &> "$LOG_FILE"',
                f'echo "[{i}/{total}] Completed"',
                "",
            ]
        
        lines += [
            'echo ""',
            'echo "' + "=" * 60 + '"',
            f'echo "All {total} runs completed."',
            "",
        ]
        
        script_content = "\n".join(lines)
        script_path = self.directories["scripts"] / "run_all.sh"
        
        if not self.dry_run:
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            print(f"\u2713 Created bash script: {script_path.name}")
        else:
            print(f"[DRY RUN] Would create: run_all.sh")
        
        # Store the script path in every job entry (used by create_manifest)
        for job in self.jobs:
            job["script_path"] = script_path
        
        print(f"\n\u2713 Generated 1 sequential bash script covering {total} runs")

    def create_manifest(self) -> None:
        """Create job manifest file."""
        print(f"\n{'='*60}")
        print(f"Creating Job Manifest")
        print(f"{'='*60}\n")
        
        if not self.dry_run:
            self.manifest = ExperimentManifest(self.experiment_dir)
            
            # Set metadata
            self.manifest.data["experiment_name"] = self.experiment_name
            self.manifest.data["created_at"] = format_timestamp()
            self.manifest.data["model_names"] = self.model_names
            self.manifest.data["attack_types"] = self.attack_types
            self.manifest.data["datasets"] = self.datasets
            self.manifest.data["num_queries"] = self.num_queries
            self.manifest.data["overrides"] = self.overrides
            self.manifest.data["script_mode"] = self.script_mode
            
            # Add jobs
            for job in self.jobs:
                job_id = job["config_name"]
                self.manifest.add_job(
                    job_id=job_id,
                    model_name=job["model"],
                    attack_type=job["attack"],
                    config_path=str(job["config_path"].relative_to(self.experiment_dir)),
                    script_path=str(job["script_path"].relative_to(self.experiment_dir))
                )
                # Annotate each job entry with the dataset it targets
                self.manifest.data["jobs"][-1]["dataset"] = job.get("dataset", self.datasets[0])
            
            self.manifest.save()
            print(f"✓ Created manifest: job_manifest.json")
            print(f"  Total jobs: {len(self.jobs)}")
        else:
            print(f"[DRY RUN] Would create: job_manifest.json")
            print(f"  Total jobs: {len(self.jobs)}")
    
    def create_submission_script(self) -> None:
        """Create batch submission script."""
        print(f"\n{'='*60}")
        print(f"Creating Submission Script")
        print(f"{'='*60}\n")
        
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated batch submission script",
            f"# Experiment: {self.experiment_name}",
            f"# Generated: {format_timestamp()}",
            "#",
            "# Submits jobs in batches of 3 with 1-hour sleep intervals",
            "# to avoid queue deprioritization",
            "",
            f"echo \"Submitting jobs for experiment: {self.experiment_name}\"",
            f"echo \"Total jobs: {len(self.jobs)}\"",
            "echo \"Batch size: 3 jobs, 1-hour sleep between batches\"",
            "echo \"" + "="*60 + "\"",
            ""
        ]
        
        # Submit jobs in batches of 3
        batch_size = 3
        for i, job in enumerate(self.jobs, 1):
            script_path = job["script_path"]
            rel_path = script_path.relative_to(self.experiment_dir)
            
            # Add batch header
            if (i - 1) % batch_size == 0:
                batch_num = ((i - 1) // batch_size) + 1
                script_lines.append(f"echo \"\"")
                script_lines.append(f"echo \"Batch {batch_num} (jobs {i}-{min(i + batch_size - 1, len(self.jobs))})\"",)
                script_lines.append("echo \"" + "-"*60 + "\"")
            
            script_lines.append(f"echo \"Submitting [{i}/{len(self.jobs)}]: {job['config_name']}\"")
            script_lines.append(f"qsub {rel_path}")
            
            # Add sleep after every 3rd job (except the last job)
            if i % batch_size == 0 and i < len(self.jobs):
                script_lines.append("")
                script_lines.append(f"echo \"\"")
                script_lines.append(f"echo \"Sleeping for 1 hour before next batch...\"")
                script_lines.append(f"echo \"Next batch starts at: $(date -d '+1 hour' '+%Y-%m-%d %H:%M:%S')\"")
                script_lines.append("sleep 3600")
                script_lines.append(f"echo \"Resuming submissions at $(date '+%Y-%m-%d %H:%M:%S')\"")
            
            script_lines.append("")
        
        script_lines.extend([
            "echo \"" + "="*60 + "\"",
            f"echo \"Submitted all {len(self.jobs)} jobs\"",
            "echo \"Check status with: qstat -u $USER\"",
            f"echo \"Check running jobs: ls {self.experiment_dir.name}/*_running.txt\"",
            ""
        ])
        
        script_content = "\n".join(script_lines)
        submit_script_path = self.experiment_dir / "submit_all.sh"
        
        if not self.dry_run:
            submit_script_path.write_text(script_content)
            submit_script_path.chmod(0o755)
            print(f"✓ Created submission script: submit_all.sh")
        else:
            print(f"[DRY RUN] Would create: submit_all.sh")
    
    def print_summary(self) -> None:
        """Print summary of generated experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment Generation {'Complete' if not self.dry_run else 'Summary (DRY RUN)'}")
        print(f"{'='*60}\n")
        
        print(f"Experiment: {self.experiment_dir}")
        print(f"\nGenerated Files:")
        print(f"  - {len(self.jobs)} config files in configs/")
        if self.script_mode == "bash":
            print(f"  - 1 sequential bash script: scripts/run_all.sh")
        else:
            print(f"  - {len(self.jobs)} PBS scripts in scripts/")
            print(f"  - submit_all.sh")
        print(f"  - job_manifest.json")
        
        print(f"\nExperiment Details:")
        print(f"  Models: {', '.join(self.model_names)}")
        print(f"  Attack Types: {len(self.attack_types)}")
        print(f"  Datasets: {', '.join(self.datasets)}")
        print(f"  Script Mode: {self.script_mode}")
        print(f"  Total Jobs: {len(self.jobs)}")
        
        if self.num_queries:
            print(f"  Queries per Job: {self.num_queries}")
        
        # Resource summary
        num_queries = self.num_queries if self.num_queries else 30
        total_gpus = sum(
            get_job_resources(job["model"], job["attack"], num_queries)["ngpus"]
            for job in self.jobs
        )
        max_walltime = max(
            get_job_resources(job["model"], job["attack"], num_queries)["walltime_hours"]
            for job in self.jobs
        )
        
        print(f"\nResource Requirements:")
        print(f"  Total GPU-hours: {total_gpus * max_walltime}")
        print(f"  Max walltime: {max_walltime} hours")
        
        if not self.dry_run:
            print(f"\n{'='*60}")
            print(f"Next Steps:")
            print(f"{'='*60}\n")
            print(f"1. Review generated configs in:")
            print(f"   {self.directories['configs']}")
            if self.script_mode == "bash":
                print(f"\n2. Run all experiments sequentially:")
                print(f"   bash {self.experiment_dir}/scripts/run_all.sh")
                print(f"\n3. After completion, aggregate results:")
            else:
                print(f"\n2. Submit jobs:")
                print(f"   cd {self.experiment_dir}")
                print(f"   bash submit_all.sh")
                print(f"\n3. Check job status:")
                print(f"   qstat -u $USER")
                print(f"\n4. After completion, aggregate results:")
            print(f"   python orchestrator/results_aggregator.py \\")
            print(f"       --experiment_dir {self.experiment_dir} \\")
            print(f"       --output_file results.csv")
        
        print()
    
    def run(self) -> None:
        """Execute the full orchestration workflow."""
        try:
            # Validate inputs
            print("Validating inputs...")
            self.validate_inputs()
            print("✓ All inputs valid")
            
            # Setup directories
            self.setup_directories()
            
            # Generate configs
            self.generate_configs()
            
            # Generate job scripts
            if self.script_mode == "pbs":
                self.generate_pbs_scripts()
            else:  # bash
                self.generate_bash_script()
            
            # Create manifest
            self.create_manifest()
            
            # Create submission script (PBS mode only)
            if self.script_mode == "pbs":
                self.create_submission_script()
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ Error during orchestration:")
            print(f"   {str(e)}")
            if "--debug" in sys.argv:
                print("\nFull traceback:")
                traceback.print_exc()
            sys.exit(1)


def expand_all_keyword(items: List[str], available_items: Dict[str, Any]) -> List[str]:
    """
    Expand 'all' keyword to all available items.
    
    Args:
        items: List of item names (may include 'all')
        available_items: Dictionary of available items
        
    Returns:
        Expanded list of items
    """
    if "all" in items:
        # Replace 'all' with all available items
        return list(available_items.keys())
    return items


def parse_overrides(override_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse override string into dictionary.
    
    Args:
        override_str: JSON string with overrides
        
    Returns:
        Dictionary of overrides
    """
    if not override_str:
        return {}
    
    try:
        return json.loads(override_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in overrides: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment Orchestrator - Generate configs and PBS/bash scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (PBS scripts, single dataset)
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "baseline_attacks" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" "ecs_gcg_suffix"
  
  # Single sequential bash script instead of PBS scripts
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "bash_run" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" "ecs_gcg_suffix" \\
      --script_mode bash
  
  # Multiple datasets in one sweep
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "multi_dataset" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" \\
      --dataset failsafeqa anah faitheval
  
  # Use 'all' for all datasets
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "all_datasets" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" \\
      --dataset all --script_mode bash
  
  # Use 'all' for all attack types
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "all_attacks_test" \\
      --model_names "llama3.2-1b" \\
      --attack_types all \\
      --num_queries 10
  
  # Multiple models and attacks
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "comprehensive_test" \\
      --model_names "llama3.2-1b" "llama3-8b" \\
      --attack_types "vanilla_gcg_suffix" "reinforce_gcg_synonym" \\
      --num_queries 50
  
  # With custom overrides
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "custom_params" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" \\
      --overrides '{"gcg.common.num_steps": 200}'
  
  # Dry run (no files created)
  python orchestrator/experiment_orchestrator.py \\
      --experiment_name "test" \\
      --model_names "llama3.2-1b" \\
      --attack_types "vanilla_gcg_suffix" \\
      --dry_run
        """
    )
    
    # Required arguments (unless using --list commands)
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Descriptive name for this experiment batch"
    )
    
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        help=f"Model names to test. Use 'all' for all models. Available: {list(MODEL_NAMES.keys())}"
    )
    
    parser.add_argument(
        "--attack_types",
        type=str,
        nargs="+",
        help=f"Attack types to run. Use 'all' for all types. Available: {list(ATTACK_TYPES.keys())}"
    )
    
    # Optional arguments
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        type=str,
        nargs="+",
        default=["failsafeqa"],
        help=f"One or more dataset names. Use 'all' for all datasets. "
             f"Available: {list(DATASET_NAMES.keys())} (default: failsafeqa)"
    )
    
    parser.add_argument(
        "--script_mode",
        type=str,
        choices=["pbs", "bash"],
        default="pbs",
        help="Script generation mode: 'pbs' for individual PBS job scripts (default); "
             "'bash' for a single sequential bash script containing all runs"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )
    
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to run (default: use method defaults)"
    )
    
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help='JSON string with config overrides, e.g., \'{"gcg.common.num_steps": 200}\''
    )
    
    parser.add_argument(
        "--conda_env",
        type=str,
        default=DEFAULT_CONDA_ENV,
        help="Conda environment name (default: $INTRINSIC_HALL_CONDA_ENV, else intrinsic-hall)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run - show what would be created without creating files"
    )
    
    parser.add_argument(
        "--list_attacks",
        action="store_true",
        help="List all available attack types and exit"
    )
    
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full error traceback on failure"
    )
    
    args = parser.parse_args()
    
    # Handle list commands (these don't require other arguments)
    if args.list_attacks:
        print("\nAvailable Attack Types:")
        print("=" * 60)
        for attack_type in sorted(ATTACK_TYPES.keys()):
            desc = get_attack_description(attack_type)
            print(f"  {attack_type:30s} - {desc}")
        print(f"\nTotal: {len(ATTACK_TYPES)} attack types")
        return
    
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 60)
        for short_name, hf_name in MODEL_NAMES.items():
            gpus = MODEL_GPU_REQUIREMENTS.get(short_name, "?")
            print(f"  {short_name:15s} ({gpus} GPU{'s' if gpus > 1 else ' '}) - {hf_name}")
        print(f"\nTotal: {len(MODEL_NAMES)} models")
        return
    
    # Validate required arguments if not using list commands
    if not args.experiment_name:
        parser.error("--experiment_name is required")
    if not args.model_names:
        parser.error("--model_names is required")
    if not args.attack_types:
        parser.error("--attack_types is required")
    
    # Expand 'all' keyword if used
    model_names = expand_all_keyword(args.model_names, MODEL_NAMES)
    attack_types = expand_all_keyword(args.attack_types, ATTACK_TYPES)
    datasets = expand_all_keyword(args.datasets, DATASET_NAMES)
    
    # Parse overrides
    overrides = parse_overrides(args.overrides)
    
    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(
        experiment_name=args.experiment_name,
        model_names=model_names,
        datasets=datasets,
        attack_types=attack_types,
        output_base_dir=args.output_base_dir,
        num_queries=args.num_queries,
        overrides=overrides,
        conda_env=args.conda_env,
        dry_run=args.dry_run,
        script_mode=args.script_mode
    )
    
    orchestrator.run()


if __name__ == "__main__":
    main()
