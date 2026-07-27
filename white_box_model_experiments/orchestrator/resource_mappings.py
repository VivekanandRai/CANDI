"""
Resource mappings for HPC cluster resource allocation.

Defines GPU, memory, and walltime requirements for different models and attack types.
"""

from typing import Dict

# ==================== GPU REQUIREMENTS ====================

# Number of GPUs required per model
MODEL_GPU_REQUIREMENTS: Dict[str, int] = {
    # Small models (1 GPU)
    "llama3.2-1b": 1,
    "gemma3-1b": 1,
    "gemma3-270m": 1,
    "qwen3-4b": 1,
    "gemma3-4b": 1,
    "llama3.2-3b": 1,
    "qwen3-1.7b": 1,
    
    # Medium models (2 GPUs)
    "llama3.1-8b": 2,
    "qwen3-8b": 2,
    "llama3-8b": 2,
    "llama2-7b": 2,
    "llama2-13b": 2,
}

# ==================== MEMORY REQUIREMENTS ====================

# Memory requirements per model (in GB)
MODEL_MEMORY_REQUIREMENTS: Dict[str, int] = {
    # Small models
    "llama3.2-1b": 32,
    "llama3.2-3b": 32,
    "llama3.1-8b": 64,
    "gemma3-1b": 32,
    "gemma3-270m": 16,
    "qwen3-4b": 32,
    "qwen3-8b": 64,
    "qwen3-1.7b": 32,
    
    # Medium models
    "llama3-8b": 64,
    "llama2-7b": 64,
    "gemma3-4b": 64,
    
    # Large models
    "llama2-13b": 128,
}

# ==================== WALLTIME REQUIREMENTS ====================

# Base number of queries for walltime estimates
BASE_NUM_QUERIES = 30

# Walltime per attack method (in hours) for BASE_NUM_QUERIES (30 queries)
# These are conservative estimates - actual time may vary
# Will be scaled proportionally based on actual number of queries
ATTACK_WALLTIME: Dict[str, int] = {
    # GCG variants (gradient-based, relatively fast)
    "vanilla_gcg": 4,
    
    # Reinforce GCG (longer due to judge model calls)
    
    # AutoDAN (genetic algorithm, moderate time)
    "vanilla_autodan": 8,
    
    # SRA (sampling-based, fast)
    "sra": 4,
    
    # PAIR (LLM-based, longest)
    "vanilla_pair": 10,
    
    # SECA (LLM-based proposals)
    "vanilla_seca": 8,
}

# ==================== CPU REQUIREMENTS ====================

# Default number of CPUs per job
DEFAULT_CPUS = 4

# ==================== HELPER FUNCTIONS ====================

def get_model_resources(model_name: str) -> Dict[str, int]:
    """
    Get resource requirements for a specific model.
    
    Args:
        model_name: Short model name (e.g., "llama3.2-1b")
        
    Returns:
        Dictionary with 'ngpus', 'memory', and 'ncpus' keys
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_GPU_REQUIREMENTS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_GPU_REQUIREMENTS.keys())}"
        )
    
    return {
        "ngpus": MODEL_GPU_REQUIREMENTS[model_name],
        "memory": MODEL_MEMORY_REQUIREMENTS[model_name],
        "ncpus": DEFAULT_CPUS
    }


def get_attack_walltime(attack_type: str, num_queries: int = BASE_NUM_QUERIES) -> int:
    """
    Get walltime requirement for a specific attack type.
    
    Args:
        attack_type: Attack type identifier (e.g., "vanilla_gcg_suffix")
        num_queries: Number of queries to run (default: 30)
        
    Returns:
        Walltime in hours, scaled based on num_queries
        
    Raises:
        ValueError: If attack_type is not recognized
    """
    # Extract base method from attack_type
    # e.g., "vanilla_gcg_suffix" -> "vanilla_gcg"
    # e.g., "sra_suffix_3" -> "sra"
    
    base_walltime = None
    
    for method_key in ATTACK_WALLTIME.keys():
        if attack_type.startswith(method_key):
            base_walltime = ATTACK_WALLTIME[method_key]
            break
    
    # Fallback: check if it contains any known method
    if base_walltime is None:
        for method_key in ATTACK_WALLTIME.keys():
            if method_key in attack_type:
                base_walltime = ATTACK_WALLTIME[method_key]
                break
    
    if base_walltime is None:
        raise ValueError(
            f"Cannot determine walltime for attack type: {attack_type}. "
            f"Known attack methods: {list(ATTACK_WALLTIME.keys())}"
        )
    
    # Scale walltime based on number of queries
    # Use ceiling to ensure we don't underestimate
    import math
    scaled_walltime = math.ceil(base_walltime * num_queries / BASE_NUM_QUERIES)
    
    # Ensure minimum of 1 hour
    return max(1, scaled_walltime)


def get_job_resources(model_name: str, attack_type: str, num_queries: int = BASE_NUM_QUERIES) -> Dict[str, int]:
    """
    Get complete resource requirements for a job.
    
    Args:
        model_name: Short model name
        attack_type: Attack type identifier
        num_queries: Number of queries to run (default: 30)
        
    Returns:
        Dictionary with 'ngpus', 'memory', 'ncpus', and 'walltime_hours' keys
    """
    resources = get_model_resources(model_name)
    resources["walltime_hours"] = get_attack_walltime(attack_type, num_queries)
    return resources


def format_walltime(hours: int) -> str:
    """
    Format walltime for PBS script.
    
    Args:
        hours: Number of hours
        
    Returns:
        PBS-formatted walltime string (HH:MM:SS)
    """
    return f"{hours}:00:00"


def format_pbs_resources(resources: Dict[str, int]) -> str:
    """
    Format resources for PBS #PBS -l line.
    
    Args:
        resources: Dictionary with resource requirements
        
    Returns:
        PBS resource string
        
    Example:
        >>> resources = {"ngpus": 2, "memory": 64, "ncpus": 4}
        >>> format_pbs_resources(resources)
        'select=1:ncpus=4:mem=64gb:ngpus=2'
    """
    ncpus = resources.get("ncpus", DEFAULT_CPUS)
    memory = resources.get("memory", 32)
    ngpus = resources.get("ngpus", 1)
    
    return f"select=1:ncpus={ncpus}:mem={memory}gb:ngpus={ngpus}"
