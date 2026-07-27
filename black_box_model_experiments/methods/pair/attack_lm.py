from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from methods.retry_utils import retry_sync
from utils_openrouter import (
    extract_message_content,
    extract_usage,
    openrouter_chat_completion,
    parse_json_response,
)


logger = logging.getLogger(__name__)


class AttackLMInterface:
    """Wrapper for the attacker model used by PAIR."""

    def __init__(
        self,
        attack_model: str,
        temperature: float = 1.0,
        max_tokens: int = 500,
        max_retries: int = 3,
        request_timeout: int = 120,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> None:
        self.attack_model = attack_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.retry_delay = retry_delay
        self.verbose = verbose

    def generate_attack_prompt(
        self,
        formatted_prompt: str,
        current_best_prompt: str,
    ) -> Dict[str, Any]:
        """Generate one adversarial prompt candidate."""
        def _attempt_once() -> Dict[str, Any]:
                response = openrouter_chat_completion(
                    model=self.attack_model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    max_retries=1,
                    retry_delay=self.retry_delay,
                )
                content = extract_message_content(response)
                parsed = parse_json_response(content)
                prompt = str(parsed.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("Attack model response missing non-empty 'prompt'")
                return {
                    "success": True,
                    "prompt": prompt,
                    "improvement": str(parsed.get("improvement", "")),
                    "usage": extract_usage(response),
                    "error": None,
                    "attempt": 1,
                }

        retry_result = retry_sync(_attempt_once)
        if retry_result.success:
            success_payload = dict(retry_result.value or {})
            success_payload["attempt"] = retry_result.attempts
            return success_payload

        if self.verbose:
            logger.warning(
                "Attack generation failed after %d attempts: %s",
                retry_result.attempts,
                retry_result.error,
            )

        return {
            "success": False,
            "prompt": current_best_prompt,
            "improvement": "Falling back to previous best prompt due to attack model failures.",
            "usage": None,
            "error": str(retry_result.error) if retry_result.error else "Unknown attack generation error.",
            "attempt": retry_result.attempts,
        }

    async def generate_attack_prompt_async(
        self,
        formatted_prompt: str,
        current_best_prompt: str,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        if semaphore is None:
            return await asyncio.to_thread(
                self.generate_attack_prompt,
                formatted_prompt,
                current_best_prompt,
            )
        async with semaphore:
            return await asyncio.to_thread(
                self.generate_attack_prompt,
                formatted_prompt,
                current_best_prompt,
            )
