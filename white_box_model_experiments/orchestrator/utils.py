"""
Utility functions for experiment orchestration.

Provides common utilities for directory management, validation,
and helper functions used across orchestration scripts.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def validate_experiment_name(name: str) -> bool:
    """
    Validate experiment name.
    
    Args:
        name: Proposed experiment name
        
    Returns:
        True if valid, False otherwise
        
    Rules:
        - Must be non-empty
        - Must contain only alphanumeric, underscore, hyphen
        - Must not start with a number
    """
    if not name:
        return False
    
    # Check valid characters
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        return False
    
    return True


def ensure_directory_structure(base_path: Path) -> Dict[str, Path]:
    """
    Create standard experiment directory structure.
    
    Args:
        base_path: Base path for the experiment
        
    Returns:
        Dictionary mapping directory names to Path objects
        
    Creates:
        base_path/
            configs/
            scripts/
            logs/
    """
    directories = {
        "base": base_path,
        "configs": base_path / "configs",
        "scripts": base_path / "scripts",
        "logs": base_path / "logs"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp for experiment naming.
    
    Args:
        dt: Datetime object (defaults to now)
        
    Returns:
        Formatted timestamp string (YYYYMMDD_HHMMSS)
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def get_next_experiment_number(base_dir: Path) -> int:
    """
    Get the next experiment number in sequence.
    
    Args:
        base_dir: Base experiments directory
        
    Returns:
        Next available experiment number
        
    Looks for directories named exp_NNN_* and returns NNN+1
    """
    if not base_dir.exists():
        return 1
    
    max_num = 0
    pattern = re.compile(r'^exp_(\d+)_')
    
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
    
    return max_num + 1


def create_experiment_directory(
    base_dir: Path,
    experiment_name: str
) -> Path:
    """
    Create a new experiment directory with sequential numbering.
    
    Args:
        base_dir: Base experiments directory
        experiment_name: User-provided experiment name
        
    Returns:
        Path to created experiment directory
        
    Creates directory named: exp_NNN_experiment_name
    """
    if not validate_experiment_name(experiment_name):
        raise ValueError(
            f"Invalid experiment name: {experiment_name}. "
            "Must start with letter and contain only alphanumeric, underscore, hyphen."
        )
    
    exp_num = get_next_experiment_number(base_dir)
    dir_name = f"exp_{exp_num:03d}_{experiment_name}"
    exp_dir = base_dir / dir_name
    
    # Create directory structure
    ensure_directory_structure(exp_dir)
    
    return exp_dir


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_environment_variables() -> Dict[str, str]:
    """
    Load required environment variables.
    
    Returns:
        Dictionary with environment variables
        
    Raises:
        EnvironmentError: If required variables are missing
    """
    required_vars = ["HF_TOKEN", "OPENROUTER_API_KEY"]
    env_vars = {}
    missing = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            env_vars[var] = value
        else:
            missing.append(var)
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    
    return env_vars


def get_config_name(model_name: str, attack_type: str) -> str:
    """
    Generate standard config filename.
    
    Args:
        model_name: Model name
        attack_type: Attack type
        
    Returns:
        Filename (without .json extension)
        
    Example:
        >>> get_config_name("llama3.2-1b", "vanilla_gcg_suffix")
        'llama3.2-1b_vanilla_gcg_suffix'
    """
    return f"{model_name}_{attack_type}"


def get_job_name(model_name: str, attack_type: str) -> str:
    """
    Generate PBS job name.
    
    Args:
        model_name: Model name
        attack_type: Attack type
        
    Returns:
        PBS job name (truncated to 15 chars if needed)
        
    PBS job names have limitations, so we abbreviate if needed.
    """
    job_name = f"{model_name}_{attack_type}"
    
    # PBS job names are typically limited to 15 characters
    if len(job_name) > 15:
        # Abbreviate attack type
        attack_abbrev = "".join([c for c in attack_type if c.isupper() or c.isdigit()])
        job_name = f"{model_name}_{attack_abbrev}"
    
    return job_name[:15]


def parse_result_directory(log_dir: Path) -> Optional[Path]:
    """
    Find overall_results.json in a log directory.
    
    Args:
        log_dir: Base log directory
        
    Returns:
        Path to overall_results.json if found, None otherwise
        
    Searches recursively for overall_results.json files.
    """
    if not log_dir.exists():
        return None
    
    # Look for overall_results.json files
    for results_file in log_dir.rglob("overall_results.json"):
        return results_file
    
    return None


def find_all_results(experiment_dir: Path) -> List[Path]:
    """
    Find all overall_results.json files in an experiment directory.
    
    Args:
        experiment_dir: Experiment directory
        
    Returns:
        List of paths to overall_results.json files
    """
    results = []
    logs_dir = experiment_dir / "logs"
    
    if logs_dir.exists():
        for results_file in logs_dir.rglob("overall_results.json"):
            results.append(results_file)
    
    return results


# Cluster-specific defaults, overridable without editing this file.
DEFAULT_CONDA_ENV = os.environ.get("INTRINSIC_HALL_CONDA_ENV", "intrinsic-hall")
DEFAULT_CONDA_HOOK = os.environ.get("INTRINSIC_HALL_CONDA_HOOK", "~/miniforge3/bin/conda")


def get_conda_activation_commands(conda_env: str = None) -> List[str]:
    """
    Get commands to activate the environment inside a generated job script.

    Both the conda installation path and the environment name are configurable
    via the INTRINSIC_HALL_CONDA_HOOK and INTRINSIC_HALL_CONDA_ENV environment
    variables, since they are specific to whichever cluster you run on.

    Args:
        conda_env: Name of conda environment (defaults to $INTRINSIC_HALL_CONDA_ENV)

    Returns:
        List of bash commands
    """
    conda_env = conda_env or DEFAULT_CONDA_ENV
    return [
        f'eval "$({DEFAULT_CONDA_HOOK} shell.bash hook)"',
        f'source activate {conda_env}'
    ]


def format_bash_array(items: List[str]) -> str:
    """
    Format a list as a bash array.
    
    Args:
        items: List of strings
        
    Returns:
        Bash array string
        
    Example:
        >>> format_bash_array(["a", "b", "c"])
        '("a" "b" "c")'
    """
    quoted = [f'"{item}"' for item in items]
    return f"({' '.join(quoted)})"


def abbreviate_attack_type(attack_type: str, max_length: int = 10) -> str:
    """
    Abbreviate attack type for display.
    
    Args:
        attack_type: Full attack type name
        max_length: Maximum length
        
    Returns:
        Abbreviated attack type
        
    Example:
        >>> abbreviate_attack_type("vanilla_gcg_suffix")
        'v_gcg_suf'
    """
    if len(attack_type) <= max_length:
        return attack_type
    
    parts = attack_type.split("_")
    
    # Abbreviate first part if it's "vanilla" or "ecs"
    if parts[0] in ["vanilla", "ecs", "pks"]:
        parts[0] = parts[0][0]
    
    # Abbreviate last part
    if len(parts) > 1:
        parts[-1] = parts[-1][:3]
    
    result = "_".join(parts)
    
    # If still too long, truncate
    if len(result) > max_length:
        result = result[:max_length]
    
    return result


class ExperimentManifest:
    """Helper class for managing experiment manifests."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.manifest_path = experiment_dir / "job_manifest.json"
        self.data = self._load_or_create()
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            return load_json(self.manifest_path)
        else:
            return {
                "experiment_name": self.experiment_dir.name,
                "created_at": format_timestamp(),
                "jobs": []
            }
    
    def add_job(
        self,
        job_id: str,
        model_name: str,
        attack_type: str,
        config_path: str,
        script_path: str
    ) -> None:
        """Add a job to the manifest."""
        job = {
            "job_id": job_id,
            "model": model_name,
            "attack_type": attack_type,
            "config_path": config_path,
            "script_path": script_path,
            "status": "pending"
        }
        self.data["jobs"].append(job)
    
    def save(self) -> None:
        """Save manifest to disk."""
        save_json(self.data, self.manifest_path)
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get list of all jobs."""
        return self.data.get("jobs", [])


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test experiment name validation
    print("\nExperiment name validation:")
    test_names = ["valid_name", "123invalid", "also-valid", "has spaces"]
    for name in test_names:
        valid = validate_experiment_name(name)
        print(f"  '{name}': {valid}")
    
    # Test timestamp formatting
    print(f"\nCurrent timestamp: {format_timestamp()}")
    
    # Test config naming
    print(f"\nConfig name: {get_config_name('llama3.2-1b', 'vanilla_gcg_suffix')}")
    print(f"Job name: {get_job_name('llama3.2-1b', 'vanilla_gcg_suffix')}")
    
    # Test attack type abbreviation
    print("\nAttack type abbreviations:")
    attacks = [
        "vanilla_gcg_suffix",
        "reinforce_gcg_synonym",
        "ecs_autodan_punctuation"
    ]
    for attack in attacks:
        print(f"  {attack} -> {abbreviate_attack_type(attack)}")
