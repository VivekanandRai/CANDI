# White-Box Experiment Orchestrator

> **Not required to reproduce the paper.** The supported entrypoint is
> `white_box_model_experiments/run.py`, which takes a single `--config` and needs none of this. The
> orchestrator is the batch-submission tooling used to run the experiment matrix on a **PBS/Torque**
> cluster; it generates configs and job scripts, then shells out to `run.py`. If you are not on a PBS
> cluster, use `scripts/generate_configs.py` (which reuses this package's config templates) and run
> `run.py` directly.
>
> Two cluster-specific settings are configurable via environment variables rather than editing code:
>
> | Variable | Default | Meaning |
> |---|---|---|
> | `INTRINSIC_HALL_CONDA_ENV` | `intrinsic-hall` | Environment name activated inside generated job scripts |
> | `INTRINSIC_HALL_CONDA_HOOK` | `~/miniforge3/bin/conda` | Path to the conda binary used for the shell hook |

This directory contains the tooling for generating white-box experiment configs, creating execution scripts, and aggregating results.

## What It Does

The orchestrator solves three practical problems:

1. Generate many experiment configs from templates instead of editing JSON manually.
2. Create PBS or bash execution scripts with model-aware resource settings.
3. Aggregate completed experiment results into CSV, JSON, or Markdown.

## Main Files

- `experiment_orchestrator.py`: main CLI for generating experiment directories, configs, and scripts
- `results_aggregator.py`: collects `overall_results.json` outputs into summary tables
- `config_templates.py`: attack-specific config generation
- `resource_mappings.py`: GPU, memory, CPU, and walltime defaults by model
- `utils.py`: naming, directory creation, manifests, and shared helpers
- `distribute_run_all.py`: optional splitter for a generated `run_all.sh` across local GPUs with tmux

## Prerequisites

Set environment variables before generating or running jobs:

```bash
export HF_TOKEN="hf_..."
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Notes:

- `HF_TOKEN` is required for loading Hugging Face models.
- `OPENROUTER_API_KEY` is required for attacks that call external judge/attack models, especially `PAIR` and `SECA`.
- You can export them in your shell, place them in a local `.env`, or load them via your cluster environment setup.

## Quick Start

### 1. Inspect available options

```bash
python orchestrator/experiment_orchestrator.py --list_attacks
python orchestrator/experiment_orchestrator.py --list_models
```

### 2. Generate an experiment

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "baseline_pair" \
  --model_names llama3.2-1b \
  --attack_types vanilla_pair \
  --dataset failsafeqa \
  --num_queries 10
```

This creates a directory like:

```text
experiments/exp_NNN_baseline_pair/
├── configs/
├── scripts/
├── logs/
├── job_manifest.json
└── submit_all.sh
```

### 3. Run the generated jobs

PBS mode:

```bash
cd experiments/exp_NNN_baseline_pair
qsub submit_all.sh
```

Bash mode:

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "baseline_pair_local" \
  --model_names llama3.2-1b \
  --attack_types vanilla_pair \
  --dataset failsafeqa \
  --script_mode bash \
  --num_queries 10

cd experiments/exp_NNN_baseline_pair_local
bash scripts/run_all.sh
```

### 4. Aggregate results

```bash
python orchestrator/results_aggregator.py \
  --experiment_dir experiments/exp_NNN_baseline_pair \
  --output_file results.csv
```

Markdown output:

```bash
python orchestrator/results_aggregator.py \
  --experiment_dir experiments/exp_NNN_baseline_pair \
  --output_file results.md \
  --format markdown
```

JSON output:

```bash
python orchestrator/results_aggregator.py \
  --experiment_dir experiments/exp_NNN_baseline_pair \
  --output_file results.json \
  --format json
```

## Common Workflows

### Generate multiple models and attacks

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "comparison_run" \
  --model_names llama3.2-1b llama3-8b \
  --attack_types vanilla_gcg_suffix ecs_gcg_suffix vanilla_pair \
  --dataset failsafeqa \
  --num_queries 50
```

### Use config overrides

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "tuned_run" \
  --model_names llama3.2-1b \
  --attack_types vanilla_gcg_suffix \
  --overrides '{"gcg.common.num_steps": 200, "gcg.common.search_width": 128}'
```

### Preview without creating files

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "preview_only" \
  --model_names llama3.2-1b \
  --attack_types vanilla_pair \
  --dry_run
```

### Aggregate multiple experiment directories

```bash
python orchestrator/results_aggregator.py \
  --experiment_dirs "experiments/exp_*" "logs" \
  --output_file all_results.csv
```

## Optional: Distribute a Generated `run_all.sh` Across Local GPUs

If you generated `--script_mode bash`, you can split the sequential script into per-GPU workers:

```bash
python orchestrator/distribute_run_all.py \
  --run_all_script experiments/exp_NNN_baseline_pair_local/scripts/run_all.sh
```

Specify GPUs explicitly if needed:

```bash
python orchestrator/distribute_run_all.py \
  --run_all_script experiments/exp_NNN_baseline_pair_local/scripts/run_all.sh \
  --gpu_ids 0,1,2,3
```

Then launch the generated tmux script:

```bash
bash experiments/exp_NNN_baseline_pair_local/scripts/run_tmux_distributed.sh
```

## Generated Outputs

After orchestration, each experiment directory contains:

- `configs/`: generated JSON configs, one per model/attack/dataset combination
- `scripts/`: generated PBS files or a bash runner
- `logs/`: run outputs written by the underlying `run.py` jobs
- `job_manifest.json`: experiment metadata and generated job list
- `submit_all.sh`: convenience launcher for PBS mode

After aggregation, you typically get:

- `results.csv`
- `results.md`
- `results.json`

depending on the requested format.

## Supported Inputs

### Datasets

The orchestrator supports:

- `failsafeqa`
- `anah`
- `faitheval`

You can pass one or more datasets with `--dataset` / `--datasets`.

### White-box model names

Use the short model names recognized by the white-box runner, for example:

- `llama3.2-1b`
- `llama3-8b`
- `llama2-7b`
- `llama2-13b`
- `gemma3-270m`
- `gemma3-1b`
- `gemma3-4b`

Use `--list_models` for the current full set.

### Attack types

Use `--list_attacks` to see the exact supported attack identifiers. This release supports six: `vanilla_gcg_synonym`, `vanilla_autodan_synonym`, `sra_suffix_3`, `sra_suffix_7`, `vanilla_pair` and `vanilla_seca`.

## Script Modes

The orchestrator supports two output modes:

- `pbs`: creates one PBS script per generated config and a `submit_all.sh`
- `bash`: creates a sequential shell script for local/manual execution

Example:

```bash
python orchestrator/experiment_orchestrator.py \
  --experiment_name "bash_demo" \
  --model_names llama3.2-1b \
  --attack_types vanilla_pair \
  --script_mode bash
```

## Aggregation Behavior

`results_aggregator.py` searches recursively for result summaries and works with:

- orchestrator-generated experiment directories under `experiments/`
- older log directories under `logs/`

It extracts experiment metadata, model and method information, timing, success rates, and other metrics from `overall_results.json` files where available.

## Tips

- Start with one model, one attack, and a small `--num_queries` to validate the full workflow.
- Use `--dry_run` before launching large sweeps.
- Prefer `--script_mode bash` for local debugging and `pbs` for cluster execution.
- Keep overrides focused and use JSON dot-path keys, for example `{"pair.common.n_iterations": 5}`.
- Aggregate after each completed sweep so you can quickly inspect failures or regressions.

## Minimal End-to-End Example

```bash
cd white_box_model_experiments

export HF_TOKEN="hf_..."
export OPENROUTER_API_KEY="sk-or-v1-..."

python orchestrator/experiment_orchestrator.py \
  --experiment_name "demo" \
  --model_names llama3.2-1b \
  --attack_types vanilla_pair \
  --dataset failsafeqa \
  --script_mode bash \
  --num_queries 5

bash experiments/exp_NNN_demo/scripts/run_all.sh

python orchestrator/results_aggregator.py \
  --experiment_dir experiments/exp_NNN_demo \
  --output_file demo_results.csv
```
