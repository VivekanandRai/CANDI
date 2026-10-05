from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from methods.seca.base_seca import BaseSECA


class VanillaSECA(BaseSECA):
    """SECA variant that optimizes PAIR judge hallucination score."""

    async def _compute_objective_async(
        self,
        input_context: str,
        query: str,
        expected_response: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        precomputed_response: Optional[str] = None,
        precomputed_hallucination_eval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        errors = []
        response = precomputed_response
        target_usage = None
        hallucination_eval = precomputed_hallucination_eval
        judge_usage = None

        if response is None:
            try:
                response, target_usage = await self._generate_target_response_async(
                    input_context=input_context,
                    query=query,
                    semaphore=semaphore,
                )
            except Exception as exc:
                errors.append(f"Target generation failed: {exc}")
                return {
                    "objective": 0.0,
                    "metadata": {
                        "hallucination_detected": False,
                        "hallucination_score": 0.0,
                        "rubric_scores": {},
                        "rubric_justifications": {},
                        "judge_justification": f"Target generation error: {exc}",
                        "response": None,
                        "hallucination_eval": {
                            "hallucination_detected": False,
                            "hallucination_score": 0.0,
                            "justification": f"Target generation error: {exc}",
                            "error": str(exc),
                        },
                    },
                    "target_usage": None,
                    "judge_usage": None,
                    "errors": errors,
                    "fatal_error": f"Target generation error: {exc}",
                }

        if hallucination_eval is None:
            hallucination_eval = await self.hallucination_judge.evaluate_async(
                context=input_context,
                query=query,
                response=response,
                correct_answer=expected_response,
                semaphore=semaphore,
            )
            judge_usage = hallucination_eval.get("usage")

        if hallucination_eval.get("error"):
            errors.append(f"Hallucination judge warning: {hallucination_eval['error']}")

        objective = float(hallucination_eval.get("hallucination_score", 0.0))
        metadata = {
            "hallucination_detected": bool(hallucination_eval.get("hallucination_detected", False)),
            "hallucination_score": objective,
            "rubric_scores": hallucination_eval.get("rubric_scores", {}),
            "rubric_justifications": hallucination_eval.get("rubric_justifications", {}),
            "judge_justification": str(hallucination_eval.get("justification", "")),
            "response": response,
            "hallucination_eval": hallucination_eval,
        }

        return {
            "objective": objective,
            "metadata": metadata,
            "target_usage": target_usage,
            "judge_usage": judge_usage,
            "errors": errors,
        }
