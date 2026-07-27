"""
Run script for hallucination attack experiments.

Usage:
    python run.py --config configs/template_config.json
"""

import argparse
import gc
import json
import os
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

# Allow `python white_box_model_experiments/run.py` from the repository root as
# well as `python run.py` from inside this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from methods.gcg.vanilla_gcg import VanillaGCG
from methods.autodan.vanilla_autodan import VanillaAutoDAN
from methods.sra.sra import SRA
from methods.pair.pair import PAIR
from methods.seca.vanilla_seca import VanillaSECA

env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)



# Model name mappings
MODEL_NAMES = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma3-270m": "google/gemma-3-270m-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-1b": "google/gemma-3-1b-it",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

# Dataset name mappings.
# Keep in sync with MODEL_DATASETS in black_box_model_experiments/utils_openrouter.py
# and DATASET_NAMES in orchestrator/config_templates.py.
DATASET_NAMES = {
    "failsafeqa": "dataset/failsafeqa_benchmark_data.json",
    "anah": "dataset/anah_benchmark_data.json",
    "faitheval": "dataset/faitheval_openended_benchmark_data.json",
}

# Method registry
METHOD_REGISTRY = {
    "gcg": {
        "vanilla_gcg": VanillaGCG,
    },
    "autodan": {
        "vanilla_autodan": VanillaAutoDAN,
    },
    "sra": {
        "sra": SRA,
    },
    "pair": {
        "vanilla_pair": PAIR,
    },
    "seca": {
        "vanilla_seca": VanillaSECA,
    },
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
    else:
        print("No GPU available. Running on CPU.")
    print("=" * 60)


def load_dataset(data_path: Path) -> list:
    """Load the QA dataset. All benchmark files are JSON arrays of records."""
    with open(data_path, "r") as f:
        qa_dataset = json.load(f)
    return qa_dataset


def load_model_and_tokenizer(hf_model_name: str):
    """Load the model and tokenizer."""
    print(f"Loading model {hf_model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        token=os.environ.get("HF_TOKEN"),
        )
    # Decoder-only models need left padding for batched generation.
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )

    # Use eager attention for compatibility
    if hasattr(model.config, "_attn_implementation") and model.config._attn_implementation == "sdpa":
        model.set_attn_implementation("eager")

    if torch.cuda.is_available():
        print(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    else:
        print("Model loaded on CPU.")

    return model, tokenizer


def run_attack(
    model,
    tokenizer,
    qa_pair: dict,
    config: Dict[str, Any],
    dataset_name: str,
) -> dict:
    """Run the attack on a single QA pair based on config."""
    
    question = qa_pair["query"]
    correct_answer = qa_pair["answer"]
    context = qa_pair["citations"]
    
    base_method = config["base_method"]
    method_name = config["method"]
    
    # Get method class
    if base_method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown base method: {base_method}")
    if method_name not in METHOD_REGISTRY[base_method]:
        raise ValueError(f"Unknown method: {method_name} for base method: {base_method}")
    
    AttackClass = METHOD_REGISTRY[base_method][method_name]
    
    # Get prompt settings
    prompt_config = config.get("prompt", {})
    system_prompt = prompt_config.get("system_prompt", "You are a helpful assistant.")
    prompt_format = prompt_config.get("prompt_format", "{context}\nQuestion:{query}{optim_str}\nAnswer:")
    
    if base_method == "gcg":
        gcg_config = config.get("gcg", {})
        common_params = gcg_config.get("common", {})
        method_params = gcg_config.get(method_name, {})

        attack_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "prompt_format": prompt_format,
            # Common params
            "top_k": common_params.get("top_k", 64),
            "num_steps": common_params.get("num_steps", 100),
            "search_width": common_params.get("search_width", 64),
            "batch_size": common_params.get("batch_size", 4),
            "optim_str_mode": common_params.get("optim_str_mode", "suffix"),
            "optim_str_length": common_params.get("optim_str_length"),
            "optim_str_constraint": common_params.get("optim_str_constraint"),
            "buffer_size": common_params.get("buffer_size", 1),
            "n_replace": common_params.get("n_replace", 1),
            "allow_non_ascii": common_params.get("allow_non_ascii", False),
            "verbose": common_params.get("verbose", False),
            "stagnation_patience": common_params.get("stagnation_patience", 20),
            "perplexity_threshold": common_params.get("perplexity_threshold"),
        }

        target = method_params.get("target", "Unable to answer")

        attack = AttackClass(**attack_kwargs)

        result = attack.run(
            input_context=context,
            query=question,
            expected_response=correct_answer,
            target=target,
        )
    elif base_method == "autodan":
        autodan_config = config.get("autodan", {})
        common_params = autodan_config.get("common", {})
        method_params = autodan_config.get(method_name, {})

        # AutoDAN prompt format must not contain {optim_str}; remove if present for compatibility
        if "{optim_str}" in prompt_format:
            prompt_format = prompt_format.replace("{optim_str}", "")

        attack_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "prompt_format": prompt_format,
            "optim_str_mode": common_params.get("optim_str_mode", "suffix"),
            "num_steps": common_params.get("num_steps", 100),
            "batch_size": common_params.get("batch_size", 256),
            "num_elites": common_params.get("num_elites", 13),
            "crossover_rate": common_params.get("crossover_rate", 0.5),
            "mutation_rate": common_params.get("mutation_rate", 0.01),
            "num_points": common_params.get("num_points", 5),
            "hga_interval": common_params.get("hga_interval", 5),
            "allow_non_ascii": common_params.get("allow_non_ascii", False),
            "verbose": common_params.get("verbose", False),
            "stagnation_patience": common_params.get("stagnation_patience", 20),
            "initial_prompt_path": common_params.get("initial_prompt_path"),
            "reference_prompts": common_params.get("reference_prompts"),
        }

        attack_kwargs["target_phrase"] = method_params.get("target_phrase", "Unable to answer")

        attack = AttackClass(**attack_kwargs)

        result = attack.run(
            input_context=context,
            query=question,
            expected_response=correct_answer,
            initial_generation=method_params.get("initial_generation"),
        )
    elif base_method == "sra":
        sra_config = config.get("sra", {})
        common_params = sra_config.get("common", {})

        attack_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "prompt_format": prompt_format,
            "batch_size": common_params.get("batch_size", 128),
            "top_p": common_params.get("top_p", 0.99),
            "top_k": common_params.get("top_k", 100),
            "threshold": common_params.get("threshold", 1_000_000),
            "response_length": common_params.get("response_length", 128),
            "prompt_length_threshold": common_params.get("prompt_length_threshold", 1000),
            "stagnation_patience": common_params.get("stagnation_patience", 20),
            "use_chat_template": common_params.get("use_chat_template", False),
            "defense_type": common_params.get("defense_type"),
        }

        attack = AttackClass(**attack_kwargs)

        result = attack.run(
            input_context=context,
            query=question,
            expected_response=correct_answer,
        )
    elif base_method == "pair":
        pair_config = config.get("pair", {})
        common_params = pair_config.get("common", {})
        method_params = pair_config.get(method_name, {})

        attack_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "prompt_format": prompt_format,
            "n_streams": common_params.get("n_streams", 5),
            "n_iterations": common_params.get("n_iterations", 10),
            "keep_last_n": common_params.get("keep_last_n", 3),
            "max_retries": common_params.get("max_retries", 3),
            "attack_temp": common_params.get("attack_temp", 1.0),
            "attack_model": method_params.get("attack_model", "google/gemini-2.5-flash-lite"),
            "judge_model": method_params.get("judge_model", "google/gemini-2.5-flash-lite"),
            "semantic_similarity_threshold": common_params.get("semantic_similarity_threshold", 0.7),
            "hallucination_score_threshold": common_params.get("hallucination_score_threshold", 5.0),
            "verbose": common_params.get("verbose", False),
        }

        attack = AttackClass(**attack_kwargs)

        result = attack.run(
            input_context=context,
            query=question,
            expected_response=correct_answer,
        )
    elif base_method == "seca":
        seca_config = config.get("seca", {})
        common_params = seca_config.get("common", {})
        method_params = seca_config.get(method_name, {})

        # Remove {optim_str} if present since SECA doesn't use it
        if "{optim_str}" in prompt_format:
            prompt_format = prompt_format.replace("{optim_str}", "")

        attack_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "prompt_format": prompt_format,
            "top_N_most_adversarial": common_params.get("top_N_most_adversarial", 5),
            "candidate_size_M": common_params.get("candidate_size_M", 3),
            "max_iteration": common_params.get("max_iteration", 10),
            "termination_threshold": common_params.get("termination_threshold"),
            "semantic_proposer_model": common_params.get("semantic_proposer_model", "openai/gpt-4o-mini"),
            "feasibility_evaluator_model": common_params.get("feasibility_evaluator_model", "openai/gpt-4o-mini"),
            "stagnation_patience": common_params.get("stagnation_patience", 5),
            "verbose": common_params.get("verbose", False),
        }

        target = method_params.get("target", "Unable to answer based on given passages.")
        attack_kwargs["target"] = target

        attack = AttackClass(**attack_kwargs)

        result = attack.run(
            input_context=context,
            query=question,
            expected_response=correct_answer,
            target=target,
        )
    else:
        raise ValueError(f"Unsupported base method: {base_method}")
    
    # Compute coherence and similarity metrics for this attack
    try:
        adversarial_query = result["adversarial_query"]
        coherence_metrics = attack.get_coherence_and_similarity(
            result_dict=result,
            original_query=question,
            adversarial_query=adversarial_query,
        )
        result.update(coherence_metrics)
    except Exception as e:
        import traceback
        error_msg = f"Error computing coherence metrics: {type(e).__name__}: {str(e)}"
        print(f"Warning: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        result["coherence_metrics_error"] = error_msg
        result["coherence_metrics_traceback"] = traceback.format_exc()
    
    # Clean up attack object to free memory
    del attack
    torch.cuda.empty_cache()
    
    return result



def main():
    parser = argparse.ArgumentParser(description="Run hallucination attack experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the config, resolve the dataset and output paths, print the "
            "resolved plan and exit. Requires no GPU and no API keys."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO). DEBUG is very noisy.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Print GPU info
    print_gpu_info()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    print(f"Base method: {config['base_method']}, Method: {config['method']}")

    # Get model name from config
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("Config must specify 'model_name'")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: {list(MODEL_NAMES.keys())}")
    
    hf_model_name = MODEL_NAMES[model_name]

    # Get dataset name from config
    dataset_name = config.get("dataset")
    if not dataset_name:
        raise ValueError("Config must specify 'dataset'")
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_NAMES.keys())}")
    
    dataset_path = DATASET_NAMES[dataset_name]
    
    # Validate the method registry entry before doing any expensive work.
    base_method = config.get("base_method")
    if base_method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown base_method: {base_method}. Supported: {list(METHOD_REGISTRY.keys())}"
        )
    if config.get("method") not in METHOD_REGISTRY[base_method]:
        raise ValueError(
            f"Unknown method: {config.get('method')} for base_method {base_method}. "
            f"Supported: {list(METHOD_REGISTRY[base_method].keys())}"
        )

    # Set up paths
    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir.parent / dataset_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Run `python scripts/prepare_failsafeqa.py` if you have not prepared the data yet."
        )

    # Create results directory with method/model/timestamp structure
    method_name = config["method"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_timestamp_dir = f"{model_name}_{timestamp}"
    
    if config.get("output_dir"):
        # output_dir is the base (e.g., experiments/exp_NNN/logs/method_name)
        # Always append model_timestamp for uniqueness
        base_dir = Path(config["output_dir"])
        results_dir = base_dir / model_timestamp_dir
    else:
        # Default: use project logs directory
        results_dir = script_dir / "logs" / method_name / model_timestamp_dir
    
    if args.dry_run:
        qa_dataset = load_dataset(data_path)
        print("=" * 60)
        print("DRY RUN - configuration resolved successfully, no attack executed")
        print("=" * 60)
        print(f"  config          : {config_path}")
        print(f"  model           : {model_name} -> {hf_model_name}")
        print(f"  dataset         : {dataset_name} -> {data_path}")
        print(f"  dataset records : {len(qa_dataset)}")
        print(f"  attack          : {config['base_method']} / {config['method']}")
        print(f"  num_queries     : {config.get('num_queries', len(qa_dataset))}")
        print(f"  results dir     : {results_dir}")
        print("=" * 60)
        return 0

    os.makedirs(results_dir, exist_ok=True)

    # Save config to results directory
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    qa_dataset = load_dataset(data_path)
    print(f"Loaded {len(qa_dataset)} QA pairs.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(hf_model_name)

    # Get number of queries to run
    num_queries = config.get("num_queries", len(qa_dataset))

    # Run attacks
    result_list = []
    failed_queries = 0

    for i, qa_pair in enumerate(qa_dataset[:num_queries]):
        print(f"\n{'='*60}")
        print(f"Query {i + 1}/{num_queries}")
        print(f"Question: {qa_pair['query'][:100]}...")
        print(f"Context length: {len(qa_pair['citations'])} chars")
        print("=" * 60)

        try:
            start_time = datetime.now()

            result = run_attack(
                model=model,
                tokenizer=tokenizer,
                qa_pair=qa_pair,
                config=config,
                dataset_name=dataset_name,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create result entry
            adversarial_query = result["adversarial_query"]
            best_suffix = result.get("best_string")
            if best_suffix is None and adversarial_query.startswith(qa_pair["query"]):
                best_suffix = adversarial_query[len(qa_pair["query"]):]

            result_entry = {
                "query_index": i,
                "original_question": qa_pair["query"],
                "expected_answer": qa_pair["answer"],
                "adversarial_query": adversarial_query,
                "best_suffix": best_suffix,
                "best_loss": result.get("best_loss"),
                "steps": result["steps"],
                "initial_response": result.get("initial_response", ""),
                "initial_hallucination": result["initial_hallucination"],
                "initial_hallucination_justification": result.get("initial_hallucination_justification", ""),
                "final_response": result.get("final_adversarial_response", ""),
                "final_hallucination": result["final_hallucination"],
                "final_hallucination_justification": result.get("final_hallucination_justification", ""),
                "stagnation_triggered": result.get("stagnation_triggered", False),
                "time_taken_seconds": duration,
                "losses_history": result.get("losses", []),
                "suffix_history": result.get("strings", []),
                # Add similarity and coherence metrics
                "cosine_similarity": result.get("cosine_similarity"),
                "semantic_equivalence": result.get("semantic_equivalence"),
                "adversarial_query_perplexity": result.get("adversarial_query_perplexity"),
                "rouge_scores": result.get("rouge_scores"),
                "coherence_metrics_error": result.get("coherence_metrics_error"),
                "coherence_metrics_traceback": result.get("coherence_metrics_traceback"),
            }

            result_list.append(result_entry)

            # Save individual result
            with open(results_dir / f"result_{i}.json", "w") as f:
                json.dump(result_entry, f, indent=4)

            print(f"Finished query {i + 1}.")
            if "best_string" in result:
                print(f"Best suffix: {result.get('best_string')}")
                print(f"Best loss: {result.get('best_loss')}")
            print(f"Initial hallucination: {result['initial_hallucination']}")
            print(f"Final hallucination: {result['final_hallucination']}")
            print(f"Time taken: {duration:.2f}s")
            
            # Explicit cleanup after successful iteration
            del result
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError as e:
            failed_queries += 1
            print(f"CUDA OOM during attack: {e}")
            print(f"Memory before cleanup - Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            
            # Aggressive cleanup
            traceback.clear_frames(e.__traceback__)
            del e
            result = None
            adversarial_query = None
            
            # Clear CUDA cache multiple times
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Memory after cleanup - Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            # raise
            continue

        except Exception as e:
            failed_queries += 1
            print(f"Error during attack: {e}")
            traceback.print_exc()

            # Cleanup on general exception too
            result = None
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Compute summary statistics
    if result_list:
        initial_correct = sum(not r["initial_hallucination"] for r in result_list)
        final_correct = sum(not r["final_hallucination"] for r in result_list)
        attack_success_count = sum(
            r["initial_hallucination"] == False and r["final_hallucination"] == True
            for r in result_list
        )

        initial_accuracy = initial_correct / len(result_list)
        final_accuracy = final_correct / len(result_list)
        attack_success_rate = attack_success_count / initial_correct if initial_correct > 0 else 0.0

        # Get config parameters for summary, from the section belonging to the
        # attack that actually ran (previously always read config["gcg"]["common"],
        # so non-GCG runs reported empty/default hyperparameters).
        common_params = config.get(config["base_method"], {}).get("common", {})

        # Aggregate coherence and similarity metrics using mean
        coherence_aggregates = {
            "cosine_similarity": [],
            "semantic_equivalence_is_equivalent": [],
            "adversarial_query_perplexity": [],
            "rouge1_f1": [],
            "rouge2_f1": [],
            "rougeL_f1": [],
        }

        for result_entry in result_list:
            if "cosine_similarity" in result_entry and result_entry["cosine_similarity"] is not None:
                coherence_aggregates["cosine_similarity"].append(result_entry["cosine_similarity"])
            
            if "semantic_equivalence" in result_entry and result_entry["semantic_equivalence"] is not None:
                sem_equiv = result_entry["semantic_equivalence"]
                if isinstance(sem_equiv, dict) and "is_equivalent" in sem_equiv:
                    # Convert boolean to int for aggregation
                    is_equiv_value = sem_equiv["is_equivalent"]
                    if isinstance(is_equiv_value, bool):
                        coherence_aggregates["semantic_equivalence_is_equivalent"].append(int(is_equiv_value))
            
            if "adversarial_query_perplexity" in result_entry and result_entry["adversarial_query_perplexity"] is not None:
                coherence_aggregates["adversarial_query_perplexity"].append(result_entry["adversarial_query_perplexity"])
            
            if "rouge_scores" in result_entry and result_entry["rouge_scores"] is not None:
                rouge_scores = result_entry["rouge_scores"]
                if isinstance(rouge_scores, dict):
                    if rouge_scores.get("rouge1_f1") is not None:
                        coherence_aggregates["rouge1_f1"].append(rouge_scores["rouge1_f1"])
                    if rouge_scores.get("rouge2_f1") is not None:
                        coherence_aggregates["rouge2_f1"].append(rouge_scores["rouge2_f1"])
                    if rouge_scores.get("rougeL_f1") is not None:
                        coherence_aggregates["rougeL_f1"].append(rouge_scores["rougeL_f1"])

        # Compute mean aggregates
        coherence_means = {}
        for metric_name, values in coherence_aggregates.items():
            if values:
                coherence_means[f"mean_{metric_name}"] = sum(values) / len(values)
            else:
                coherence_means[f"mean_{metric_name}"] = None

        overall_summary = {
            "model_name": model_name,
            "hf_model_name": hf_model_name,
            "base_method": config["base_method"],
            "method": config["method"],
            "config_path": str(config_path),
            "dataset_path": str(data_path),
            "num_optim_ids": common_params.get("optim_str_length"),
            "num_steps": common_params.get("num_steps", 100),
            "search_width": common_params.get("search_width", 64),
            "batch_size": common_params.get("batch_size", 4),
            "stagnation_patience": common_params.get("stagnation_patience", 20),
            "prompt_length_threshold": common_params.get("prompt_length_threshold"),
            "total_queries": len(result_list),
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "attack_success_count": attack_success_count,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "attack_success_rate": attack_success_rate,
            "avg_time_per_query_seconds": sum(r["time_taken_seconds"] for r in result_list) / len(result_list),
            "total_time_seconds": sum(r["time_taken_seconds"] for r in result_list),
        }

        # Add coherence metric aggregates to summary
        overall_summary.update(coherence_means)

        # Save overall results with full config
        with open(results_dir / "overall_results.json", "w") as f:
            json.dump({
                "summary": overall_summary, 
                "config": config,
                "results": result_list
            }, f, indent=4)

        print(f"\n{'='*60}")
        print("Summary")
        print("=" * 60)
        print(f"Method: {config['method']}")
        print(f"Model: {model_name}")
        print(f"Total queries: {overall_summary['total_queries']}")
        print(f"Initial accuracy: {overall_summary['initial_accuracy']:.2%}")
        print(f"Final accuracy: {overall_summary['final_accuracy']:.2%}")
        print(f"Attack success rate: {overall_summary['attack_success_rate']:.2%}")
        print(f"Avg time per query: {overall_summary['avg_time_per_query_seconds']:.2f}s")
        
        # Print coherence metrics
        print(f"\nCoherence and Similarity Metrics:")
        if coherence_means.get("mean_cosine_similarity") is not None:
            print(f"  Mean cosine similarity: {coherence_means['mean_cosine_similarity']:.4f}")
        if coherence_means.get("mean_semantic_equivalence_is_equivalent") is not None:
            print(f"  Mean semantic equivalence rate: {coherence_means['mean_semantic_equivalence_is_equivalent']:.2%}")
        if coherence_means.get("mean_adversarial_query_perplexity") is not None:
            print(f"  Mean adversarial query perplexity: {coherence_means['mean_adversarial_query_perplexity']:.4f}")
        if coherence_means.get("mean_rouge1_f1") is not None:
            print(f"  Mean ROUGE-1 F1: {coherence_means['mean_rouge1_f1']:.4f}")
        if coherence_means.get("mean_rouge2_f1") is not None:
            print(f"  Mean ROUGE-2 F1: {coherence_means['mean_rouge2_f1']:.4f}")
        if coherence_means.get("mean_rougeL_f1") is not None:
            print(f"  Mean ROUGE-L F1: {coherence_means['mean_rougeL_f1']:.4f}")
        
        if failed_queries:
            print(f"\nWARNING: {failed_queries} quer{'y' if failed_queries == 1 else 'ies'} failed and were skipped.")
        print(f"\nResults saved to {results_dir}")
        return 0

    # Every query failed. Exit non-zero so callers and job schedulers notice,
    # rather than reporting a silent success.
    print("\nNo results to summarize: every query failed.")
    print(f"Config: {config_path}. See the traceback(s) above for the cause.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
