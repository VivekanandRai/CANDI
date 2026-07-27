from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ConversationManager:
    """Tracks PAIR conversation streams and best prompts."""

    def __init__(self, n_streams: int = 5, keep_last_n: int = 3) -> None:
        self.n_streams = n_streams
        self.keep_last_n = keep_last_n

        self.conversations: List[List[Dict[str, str]]] = []
        self.best_prompts: List[str] = []
        self.best_scores: List[float] = []
        self.iteration_counts: List[int] = []

    def initialize(self, original_query: str, expected_response: str, system_prompt: str) -> None:
        self.conversations = []
        self.best_prompts = []
        self.best_scores = []
        self.iteration_counts = []

        for _ in range(self.n_streams):
            history = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Original Query: {original_query}\n"
                        f"Expected Correct Answer: {expected_response}\n"
                        "Generate an adversarial variant of the query that can induce hallucination."
                    ),
                },
            ]
            self.conversations.append(history)
            self.best_prompts.append(original_query)
            self.best_scores.append(0.0)
            self.iteration_counts.append(0)

    def get_formatted_prompt(self, stream_idx: int) -> str:
        if stream_idx >= len(self.conversations):
            return ""
        parts: List[str] = []
        for msg in self.conversations[stream_idx]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(parts)

    def get_best_prompt(self, stream_idx: int) -> str:
        if stream_idx >= len(self.best_prompts):
            return ""
        return self.best_prompts[stream_idx]

    def add_iteration_result(
        self,
        stream_idx: int,
        adversarial_prompt: str,
        score: float,
        feedback_dict: Dict[str, Any],
        model_response: Optional[str] = None,
    ) -> None:
        if stream_idx >= len(self.conversations):
            return

        if score > self.best_scores[stream_idx]:
            self.best_scores[stream_idx] = score
            self.best_prompts[stream_idx] = adversarial_prompt

        assistant_payload = {
            "prompt": adversarial_prompt,
            "improvement": feedback_dict.get("improvement", ""),
        }
        self.conversations[stream_idx].append(
            {"role": "assistant", "content": json.dumps(assistant_payload)}
        )

        feedback_lines = [
            f"Score: {score:.2f}/10",
            f"Hallucination Detected: {feedback_dict.get('hallucination_detected')}",
            f"Hallucination Score: {feedback_dict.get('hallucination_score')}",
            f"Semantic Equivalent: {feedback_dict.get('is_semantically_equivalent')}",
            f"Hallucination Feedback: {feedback_dict.get('hallucination_justification', '')[:200]}",
            f"Similarity Feedback: {feedback_dict.get('similarity_reasoning', '')[:200]}",
        ]
        if model_response is not None:
            feedback_lines.append(f"Model Response Preview: {model_response[:160]}")

        self.conversations[stream_idx].append(
            {"role": "user", "content": "\n".join(feedback_lines)}
        )

        self.iteration_counts[stream_idx] += 1
        self._truncate(stream_idx)

    def get_all_conversations(self) -> List[List[Dict[str, str]]]:
        return self.conversations

    def get_stats(self) -> Dict[str, Any]:
        avg_score = sum(self.best_scores) / len(self.best_scores) if self.best_scores else 0.0
        max_score = max(self.best_scores) if self.best_scores else 0.0
        return {
            "n_streams": self.n_streams,
            "iterations_per_stream": self.iteration_counts,
            "best_scores": self.best_scores,
            "average_score": avg_score,
            "max_score": max_score,
            "message_counts": [len(conv) for conv in self.conversations],
        }

    def _truncate(self, stream_idx: int) -> None:
        conversation = self.conversations[stream_idx]
        target_len = 2 * self.keep_last_n + 1  # system + last N pairs
        if len(conversation) <= target_len:
            return
        self.conversations[stream_idx] = [conversation[0]] + conversation[-(target_len - 1) :]
