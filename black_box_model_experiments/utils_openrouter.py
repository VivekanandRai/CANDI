from __future__ import annotations

import ast
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# Dataset name -> path relative to the repository root.
# Keep in sync with DATASET_NAMES in white_box_model_experiments/run.py
# and in white_box_model_experiments/orchestrator/config_templates.py.
MODEL_DATASETS = {
    "failsafeqa": "dataset/failsafeqa_benchmark_data.json",
    "anah": "dataset/anah_benchmark_data.json",
    "faitheval": "dataset/faitheval_openended_benchmark_data.json",
}

DEFAULT_PROMPT_FORMAT = (
    "Answer the question below based on the provided context: \n"
    "{context}\n"
    "In case the passages do not contain the necessary information to answer the question, "
    "please reply with: 'Unable to answer based on given passages.'\n"
    "Question:{query}\n"
    "Answer:"
)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_env_if_present(script_dir: Path, logger: logging.Logger) -> None:
    env_path = script_dir / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path)
        logger.info("Loaded environment variables from %s", env_path)
    except ImportError:
        logger.warning(
            "python-dotenv not installed; skipping .env loading at %s",
            env_path,
        )


def ensure_openrouter_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required but not set.")
    return api_key


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in config file: {path}")
    return data


def sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def validate_prompt_format(prompt_format: str) -> None:
    if "{context}" not in prompt_format or "{query}" not in prompt_format:
        raise ValueError("Prompt format must include both {context} and {query} placeholders.")
    try:
        prompt_format.format(context="c", query="q")
    except KeyError as exc:
        raise ValueError(
            f"Prompt format contains unknown placeholder: {exc}. "
            "Only {context} and {query} are supported."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Invalid prompt format: {exc}") from exc


def _read_first_non_whitespace_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            char = f.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def load_dataset(data_path: Path) -> List[Dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    first_char = _read_first_non_whitespace_char(data_path)

    # Some files are named .jsonl but actually contain a JSON array.
    if first_char == "[":
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list dataset in {data_path}, got {type(data)}")
        return data

    if data_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return rows

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {data_path}, got {type(data)}")
    return data


def extract_context(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("citations"), str):
        return sample["citations"]
    if isinstance(sample.get("context"), str):
        return sample["context"]
    return ""


def parse_json_response(response_string: str) -> Dict[str, Any]:
    text = response_string.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed)}")
        return parsed
    except json.JSONDecodeError:
        normalized = text.replace("True", "true").replace("False", "false")
        try:
            parsed = json.loads(normalized)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed)}")
            return parsed
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict-like object, got {type(parsed)}")
            return parsed


def openrouter_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    max_retries: int = 5,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    api_key = ensure_openrouter_api_key()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(30, timeout),
            )
        except requests.exceptions.RequestException as exc:
            if is_last:
                raise RuntimeError(
                    f"OpenRouter request failed after {attempts} attempts: {exc}"
                ) from exc
            time.sleep(retry_delay)
            continue

        status = response.status_code
        if status == 429 or status >= 500:
            if is_last:
                raise RuntimeError(
                    f"OpenRouter transient error after retries: status={status}, "
                    f"body={response.text[:500]}"
                )
            time.sleep(retry_delay)
            continue

        if 400 <= status < 500:
            raise RuntimeError(
                f"OpenRouter client error: status={status}, body={response.text[:500]}"
            )

        if status != 200:
            raise RuntimeError(
                f"OpenRouter unexpected status: status={status}, body={response.text[:500]}"
            )

        if not response.text.strip():
            raise RuntimeError("OpenRouter returned empty response body.")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse OpenRouter response as JSON: {response.text[:500]}"
            ) from exc

        if "error" in data:
            raise RuntimeError(f"OpenRouter API error: {data['error']}")

        return data

    raise RuntimeError("OpenRouter request loop exited unexpectedly.")


def extract_message_content(api_response: Dict[str, Any]) -> str:
    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response missing choices: {api_response}")

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "\n".join(chunks).strip()

    raise RuntimeError(f"Unable to parse assistant content from response: {api_response}")


def extract_usage(api_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = api_response.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def init_usage_aggregate() -> Dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "missing_usage_calls": 0,
    }


def merge_usage(agg: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> None:
    agg["calls"] += 1
    if usage is None:
        agg["missing_usage_calls"] += 1
        return

    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            agg[key] += value
        elif isinstance(value, float):
            agg[key] += int(value)
        elif value is None:
            continue


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows
