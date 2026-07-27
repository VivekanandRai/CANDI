#!/usr/bin/env python3
"""
Distribute an orchestrator-generated run_all.sh across GPUs.

This script is intentionally standalone and non-breaking:
- It does not modify orchestrator behavior.
- It reads an existing sequential run_all.sh.
- It writes per-GPU worker scripts and a tmux launcher script.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


START_RE = re.compile(r'^echo "Starting (\d+) sequential runs for experiment: (.+)"$')
SEP_RE = re.compile(r'^echo "=+"$')
RUN_HEADER_RE = re.compile(r'^echo "\[(\d+)/(\d+)\] (.+)"$')
LOG_FILE_RE = re.compile(r'^LOG_FILE="(.+)"$')
CONFIG_ECHO_RE = re.compile(r'^echo "  Config : (.+)"$')
CONFIG_CMD_RE = re.compile(r"^    --config (.+) \\$")
RUN_DONE_RE = re.compile(r'^echo "\[(\d+)/(\d+)\] Completed"$')
ALL_DONE_RE = re.compile(r'^echo "All (\d+) runs completed\."$')

TIMESTAMP_LINE = 'TIMESTAMP=$(date +"%Y%m%d_%H%M%S")'
PYTHON_CMD_LINE = "python run.py \\"
LOG_ECHO_LINE = 'echo "  Log    : $LOG_FILE"'
REDIRECT_LINE = '    &> "$LOG_FILE"'


@dataclass
class RunBlock:
    """Single run block parsed from run_all.sh."""

    index: int
    total: int
    name: str
    config_path: str
    lines: List[str]


@dataclass
class ParsedRunAll:
    """Parsed representation of run_all.sh."""

    preamble_lines: List[str]
    total_runs: int
    experiment_name: str
    run_blocks: List[RunBlock]


def _parse_gpu_tokens(raw: str, source: str) -> List[str]:
    tokens = [token.strip() for token in raw.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError(f"No GPU ids found from {source}.")

    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"GPU id '{token}' from {source} is not numeric. "
                "Use numeric GPU ids (e.g., 0,1,2)."
            )

    if len(set(tokens)) != len(tokens):
        raise ValueError(f"Duplicate GPU ids found from {source}: {tokens}")

    return tokens


def resolve_gpu_ids(explicit_gpu_ids: Optional[str]) -> List[str]:
    """Resolve GPU ids from CLI, CUDA_VISIBLE_DEVICES, or nvidia-smi."""
    if explicit_gpu_ids:
        return _parse_gpu_tokens(explicit_gpu_ids, "--gpu_ids")

    env_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_gpu_ids:
        return _parse_gpu_tokens(env_gpu_ids, "CUDA_VISIBLE_DEVICES")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "nvidia-smi not found and CUDA_VISIBLE_DEVICES is unset. "
            "Set CUDA_VISIBLE_DEVICES or pass --gpu_ids."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        raise ValueError(
            f"Failed to detect GPUs with nvidia-smi: {stderr}. "
            "Set CUDA_VISIBLE_DEVICES or pass --gpu_ids."
        ) from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            "No GPUs detected from nvidia-smi output. "
            "Set CUDA_VISIBLE_DEVICES or pass --gpu_ids."
        )

    return _parse_gpu_tokens(",".join(lines), "nvidia-smi")


def parse_run_block(
    lines: List[str], start_idx: int, expected_total: int, expected_index: int
) -> Tuple[RunBlock, int]:
    """Parse one strict orchestrator block starting at `echo \"\"`."""
    if start_idx + 9 >= len(lines):
        raise ValueError("Malformed run_all.sh: incomplete run block.")

    block = lines[start_idx : start_idx + 10]

    if block[0] != 'echo ""':
        raise ValueError("Malformed run_all.sh: run block must start with `echo \"\"`.")

    header_match = RUN_HEADER_RE.match(block[1])
    if not header_match:
        raise ValueError(f"Malformed run header: {block[1]}")
    run_idx = int(header_match.group(1))
    run_total = int(header_match.group(2))
    run_name = header_match.group(3)

    if run_idx != expected_index:
        raise ValueError(f"Expected run index {expected_index}, found {run_idx}.")
    if run_total != expected_total:
        raise ValueError(f"Expected run total {expected_total}, found {run_total}.")

    if block[2] != TIMESTAMP_LINE:
        raise ValueError(f"Malformed TIMESTAMP line: {block[2]}")

    if not LOG_FILE_RE.match(block[3]):
        raise ValueError(f"Malformed LOG_FILE line: {block[3]}")

    config_echo_match = CONFIG_ECHO_RE.match(block[4])
    if not config_echo_match:
        raise ValueError(f"Malformed config echo line: {block[4]}")
    config_path_echo = config_echo_match.group(1)

    if block[5] != LOG_ECHO_LINE:
        raise ValueError(f"Malformed log echo line: {block[5]}")

    if block[6] != PYTHON_CMD_LINE:
        raise ValueError(f"Malformed python command line: {block[6]}")

    config_cmd_match = CONFIG_CMD_RE.match(block[7])
    if not config_cmd_match:
        raise ValueError(f"Malformed --config line: {block[7]}")
    config_path_cmd = config_cmd_match.group(1)

    if config_path_cmd != config_path_echo:
        raise ValueError(
            "Config path mismatch between echo and --config line: "
            f"{config_path_echo} vs {config_path_cmd}"
        )

    if block[8] != REDIRECT_LINE:
        raise ValueError(f"Malformed redirect line: {block[8]}")

    done_match = RUN_DONE_RE.match(block[9])
    if not done_match:
        raise ValueError(f"Malformed run completion line: {block[9]}")
    done_idx = int(done_match.group(1))
    done_total = int(done_match.group(2))
    if done_idx != run_idx or done_total != run_total:
        raise ValueError("Run completion index/total does not match run header.")

    run_block = RunBlock(
        index=run_idx,
        total=run_total,
        name=run_name,
        config_path=config_path_cmd,
        lines=block,
    )
    return run_block, start_idx + 10


def parse_run_all_script(script_path: Path) -> ParsedRunAll:
    """Strictly parse an orchestrator-generated run_all.sh."""
    if not script_path.exists():
        raise ValueError(f"run_all script not found: {script_path}")

    lines = script_path.read_text().splitlines()
    if not lines:
        raise ValueError("run_all.sh is empty.")
    if lines[0] != "#!/bin/bash":
        raise ValueError("run_all.sh must start with `#!/bin/bash`.")

    start_idx = None
    start_match = None
    for idx, line in enumerate(lines):
        match = START_RE.match(line)
        if match:
            start_idx = idx
            start_match = match
            break

    if start_idx is None or start_match is None:
        raise ValueError("Could not find run start marker in run_all.sh.")
    if start_idx + 1 >= len(lines) or not SEP_RE.match(lines[start_idx + 1]):
        raise ValueError("Malformed run_all.sh: missing separator after start marker.")

    total_runs = int(start_match.group(1))
    experiment_name = start_match.group(2)
    preamble = lines[:start_idx]

    blocks: List[RunBlock] = []
    idx = start_idx + 2
    expected_index = 1

    while idx < len(lines):
        if lines[idx] == 'echo ""':
            if idx + 2 < len(lines) and SEP_RE.match(lines[idx + 1]) and ALL_DONE_RE.match(lines[idx + 2]):
                break
            run_block, idx = parse_run_block(lines, idx, total_runs, expected_index)
            blocks.append(run_block)
            expected_index += 1
            while idx < len(lines) and lines[idx] == "":
                if idx + 2 < len(lines) and SEP_RE.match(lines[idx + 1]) and ALL_DONE_RE.match(lines[idx + 2]):
                    break
                idx += 1
            continue

        if lines[idx].strip() == "":
            idx += 1
            continue

        raise ValueError(f"Unexpected line while parsing run blocks: {lines[idx]}")

    if idx + 2 >= len(lines):
        raise ValueError("Malformed run_all.sh: missing final summary tail.")
    if lines[idx] != 'echo ""' or not SEP_RE.match(lines[idx + 1]):
        raise ValueError("Malformed run_all.sh: invalid tail prefix.")

    all_done_match = ALL_DONE_RE.match(lines[idx + 2])
    if not all_done_match:
        raise ValueError("Malformed run_all.sh: missing final completion summary.")
    if int(all_done_match.group(1)) != total_runs:
        raise ValueError("Tail total runs does not match start marker.")

    for tail_line in lines[idx + 3 :]:
        if tail_line.strip():
            raise ValueError(f"Unexpected trailing non-empty line: {tail_line}")

    if len(blocks) != total_runs:
        raise ValueError(
            f"Parsed {len(blocks)} run blocks, but start marker declares {total_runs}."
        )

    return ParsedRunAll(
        preamble_lines=preamble,
        total_runs=total_runs,
        experiment_name=experiment_name,
        run_blocks=blocks,
    )


def assign_round_robin(run_blocks: List[RunBlock], gpu_ids: List[str]) -> Dict[str, List[RunBlock]]:
    """Assign run blocks to GPUs in round-robin order."""
    assignments: Dict[str, List[RunBlock]] = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, run_block in enumerate(run_blocks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        assignments[gpu_id].append(run_block)
    return assignments


def build_worker_script(
    parsed: ParsedRunAll,
    gpu_id: str,
    assigned_blocks: List[RunBlock],
    num_gpus: int,
) -> str:
    """Build per-GPU worker script content."""
    lines: List[str] = []
    lines.extend(parsed.preamble_lines)
    if lines and lines[-1] != "":
        lines.append("")

    lines.extend(
        [
            f'export CUDA_VISIBLE_DEVICES="{gpu_id}"',
            'echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
            f'echo "Worker GPU {gpu_id}: {len(assigned_blocks)} run(s) assigned across {num_gpus} GPU(s)."',
            'echo "============================================================"',
            "",
        ]
    )

    if not assigned_blocks:
        lines.extend(
            [
                'echo "No runs assigned to this worker. Exiting."',
                'echo "============================================================"',
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    for run_block in assigned_blocks:
        lines.extend(run_block.lines)
        lines.append("")

    lines.extend(
        [
            'echo ""',
            'echo "============================================================"',
            f'echo "GPU {gpu_id} worker completed {len(assigned_blocks)} run(s)."',
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_tmux_launcher(gpu_ids: List[str], worker_files: List[str]) -> str:
    """Build launcher script that starts a tmux session per worker."""
    worker_array = " ".join(f'"{name}"' for name in worker_files)
    gpu_array = " ".join(f'"{gpu_id}"' for gpu_id in gpu_ids)

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "if ! command -v tmux >/dev/null 2>&1; then",
        '    echo "ERROR: tmux is not installed or not in PATH."',
        "    exit 1",
        "fi",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"',
        'EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"',
        'TMUX_LOG_DIR="$EXPERIMENT_DIR/logs/tmux"',
        "mkdir -p \"$TMUX_LOG_DIR\"",
        "",
        'TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"',
        "",
        f"worker_scripts=({worker_array})",
        f"gpu_ids=({gpu_array})",
        "",
        'echo "Launching distributed tmux sessions..."',
        'echo "Experiment: $EXPERIMENT_NAME"',
        'echo "Workers: ${#worker_scripts[@]}"',
        'echo "TMUX logs: $TMUX_LOG_DIR"',
        'echo "============================================================"',
        "",
        'for i in "${!worker_scripts[@]}"; do',
        '    worker_script="${worker_scripts[$i]}"',
        '    gpu_id="${gpu_ids[$i]}"',
        '    session_name="${EXPERIMENT_NAME}_gpu${gpu_id}_${TIMESTAMP}"',
        '    worker_path="$SCRIPT_DIR/$worker_script"',
        '    log_file="$TMUX_LOG_DIR/${session_name}.log"',
        "",
        '    if [ ! -f "$worker_path" ]; then',
        '        echo "WARNING: Missing worker script: $worker_path (skipping)"',
        "        continue",
        "    fi",
        "",
        '    echo "Starting session $session_name (GPU $gpu_id)..."',
        '    tmux new-session -d -s "$session_name" "bash \\"$worker_path\\" 2>&1 | tee \\"$log_file\\""',
        "done",
        "",
        'echo "============================================================"',
        'echo "All tmux sessions launched."',
        'echo "List sessions: tmux ls"',
        'echo "Attach: tmux attach -t <session_name>"',
        'echo "Tail logs: tail -f $TMUX_LOG_DIR/<session_name>.log"',
        "",
    ]
    return "\n".join(lines)


def write_executable(path: Path, content: str) -> None:
    """Write content to file and make it executable."""
    path.write_text(content)
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Distribute an orchestrator-generated run_all.sh across GPUs and "
            "generate per-GPU worker scripts plus a tmux launcher."
        )
    )
    parser.add_argument(
        "--run_all_script",
        type=str,
        required=True,
        help="Path to an existing orchestrator-generated run_all.sh",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids (e.g., 0,1,2). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated scripts (default: directory containing run_all.sh).",
    )

    args = parser.parse_args()

    run_all_script = Path(args.run_all_script).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_all_script.parent.resolve()
    )

    # Validate upfront so parse errors happen before writing any outputs.
    parsed = parse_run_all_script(run_all_script)
    gpu_ids = resolve_gpu_ids(args.gpu_ids)
    assignments = assign_round_robin(parsed.run_blocks, gpu_ids)

    output_dir.mkdir(parents=True, exist_ok=True)

    worker_files: List[str] = []
    for gpu_id in gpu_ids:
        worker_name = f"run_gpu_{gpu_id}.sh"
        worker_path = output_dir / worker_name
        worker_files.append(worker_name)
        worker_content = build_worker_script(
            parsed=parsed,
            gpu_id=gpu_id,
            assigned_blocks=assignments[gpu_id],
            num_gpus=len(gpu_ids),
        )
        write_executable(worker_path, worker_content)

    launcher_path = output_dir / "run_tmux_distributed.sh"
    launcher_content = build_tmux_launcher(gpu_ids=gpu_ids, worker_files=worker_files)
    write_executable(launcher_path, launcher_content)

    print("=" * 60)
    print("Distributed scripts generated successfully")
    print("=" * 60)
    print(f"Input run_all.sh : {run_all_script}")
    print(f"Experiment       : {parsed.experiment_name}")
    print(f"Total runs       : {parsed.total_runs}")
    print(f"GPU ids          : {', '.join(gpu_ids)}")
    print(f"Output directory : {output_dir}")
    print("")
    for gpu_id in gpu_ids:
        print(f"  - run_gpu_{gpu_id}.sh ({len(assignments[gpu_id])} run(s))")
    print("  - run_tmux_distributed.sh")
    print("")
    print("Next step:")
    print(f"  bash {launcher_path}")


if __name__ == "__main__":
    main()
