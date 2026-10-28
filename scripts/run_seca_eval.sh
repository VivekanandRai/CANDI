#!/usr/bin/env bash
# Run the small SECA eval configs (seca_<model>_<dataset>_eval.json) for every
# model on every dataset, one after another.
#
# Usage:
#   scripts/run_seca_eval.sh                      # all models
#   scripts/run_seca_eval.sh openai_gpt-5-nano    # only the given model(s)
#   DRY_RUN=1 scripts/run_seca_eval.sh            # validate configs, no API calls
#
# A failed run is logged and skipped; the remaining runs still execute.
# The script exits non-zero if any run failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="black_box_model_experiments/experiment_configs"
DATASETS=(anah faitheval failsafeqa)
ALL_MODELS=(
    google_gemini-2.5-flash-lite
    minimax_minimax-m2.1
    openai_gpt-5-chat
    openai_gpt-5-mini
    # openai_gpt-5-nano
)

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

total=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
step=0
failed=()

echo "[$(timestamp)] Running ${total} SECA eval runs (${#MODELS[@]} models x ${#DATASETS[@]} datasets)"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        step=$(( step + 1 ))
        config="${CONFIG_DIR}/seca_${model}_${dataset}_eval.json"

        echo
        echo "============================================================"
        echo "[$(timestamp)] Step ${step}/${total}: ${model} / ${dataset}"
        echo "============================================================"

        if [[ ! -f "$config" ]]; then
            echo "[$(timestamp)] Config not found: ${config}" >&2
            failed+=("${model}/${dataset} (missing config)")
            continue
        fi

        start=$SECONDS
        if uv run python black_box_model_experiments/run_seca.py --config "$config" "${EXTRA_ARGS[@]}"; then
            echo "[$(timestamp)] Step ${step}/${total} done in $(( SECONDS - start ))s"
        else
            code=$?
            echo "[$(timestamp)] Step ${step}/${total} failed (exit ${code}) after $(( SECONDS - start ))s" >&2
            failed+=("${model}/${dataset} (exit ${code})")
        fi
    done
done

echo
echo "============================================================"
echo "[$(timestamp)] Finished: $(( total - ${#failed[@]} ))/${total} runs succeeded"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed runs:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
