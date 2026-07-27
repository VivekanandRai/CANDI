"""Vanilla SECA implementation using cross-entropy objective.

This variant optimizes for generating a target incorrect answer string
by minimizing cross-entropy loss on that target.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import logging

import torch

from methods.seca.base_seca import BaseSECA

logger = logging.getLogger("vanilla_seca")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VanillaSECA(BaseSECA):
    """SECA with cross-entropy objective targeting an incorrect answer string.
    
    This variant computes the negative cross-entropy loss of producing a target
    string (e.g., "Unable to answer") as the objective. Higher objective values
    indicate the model is more likely to produce the target incorrect response.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        prompt_format: str | None = None,
        system_prompt: str | None = None,
        # VanillaSECA-specific parameter
        target: str = "Unable to answer based on given passages.",
        # SECA common parameters
        top_N_most_adversarial: int = 5,
        candidate_size_M: int = 3,
        max_iteration: int = 10,
        termination_threshold: float | None = None,
        semantic_proposer_model: str = "openai/gpt-4o-mini",
        feasibility_evaluator_model: str = "openai/gpt-4o-mini",
        stagnation_patience: int = 5,
        verbose: bool = False,
    ):
        """Initialize VanillaSECA.
        
        Args:
            model: Target model to attack.
            tokenizer: Tokenizer for target model.
            prompt_format: Format string for constructing prompts.
            system_prompt: System prompt for target model.
            target: Target string to optimize for (incorrect answer).
            top_N_most_adversarial: Number of parent queries to maintain.
            candidate_size_M: Number of candidates per parent per iteration.
            max_iteration: Maximum evolutionary iterations.
            termination_threshold: Objective threshold for early termination.
            semantic_proposer_model: LLM for generating semantic equivalents.
            feasibility_evaluator_model: LLM for checking semantic equivalence.
            stagnation_patience: Iterations without improvement before stopping.
            verbose: Enable verbose logging.
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            top_N_most_adversarial=top_N_most_adversarial,
            candidate_size_M=candidate_size_M,
            max_iteration=max_iteration,
            termination_threshold=termination_threshold,
            semantic_proposer_model=semantic_proposer_model,
            feasibility_evaluator_model=feasibility_evaluator_model,
            stagnation_patience=stagnation_patience,
            verbose=verbose,
        )
        
        self.target = target
        logger.info(f"VanillaSECA initialized with target: '{target}'")
    
    def run(
        self,
        input_context: str,
        query: str,
        expected_response: str,
        target: str | None = None,
    ) -> Dict[str, Any]:
        """Run VanillaSECA attack.
        
        Args:
            input_context: Context provided to the model.
            query: Original query to attack.
            expected_response: Correct answer.
            target: Override target string (optional).
            
        Returns:
            Attack results dictionary.
        """
        if target is not None:
            self.target = target
            
        return super().run(
            input_context=input_context,
            query=query,
            expected_response=expected_response,
        )
    
    def _compute_objective(
        self,
        full_query: str,
        input_context: str,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute negative cross-entropy of target string.
        
        Higher values indicate the model is more likely to produce the target.
        
        Args:
            full_query: The complete formatted prompt.
            input_context: The context portion (unused in this objective).
            
        Returns:
            Tuple of (objective_value, metadata_dict).
        """
        # Build chat prompt
        messages = [{"role": "user", "content": full_query}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize prompt and target
        prompt_ids = self.tokenizer(
            chat_prompt, return_tensors="pt"
        ).input_ids.to(self.model.device)
        
        target_ids = self.tokenizer(
            self.target, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        
        # Concatenate so logits cover the target portion
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.generation_model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab)
            
            tgt_len = target_ids.shape[1]
            # Shift so position n-1 predicts token n for the target span
            shift_start = logits.shape[1] - tgt_len
            shift_logits = logits[:, shift_start - 1 : -1, :].contiguous()
            labels = target_ids
            
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
                reduction="mean"
            )
            
            # Objective: negative CE (higher = more likely to produce target)
            obj_value = -ce_loss.item()
            
            # Token-level probabilities for metadata
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, -1, labels.unsqueeze(-1)
            ).squeeze(-1)
            token_probs = torch.exp(token_log_probs)
            mean_target_prob = token_probs.mean().item()
        
        metadata = {
            "ce_loss": ce_loss.item(),
            "mean_target_prob": mean_target_prob,
            "target": self.target,
        }
        
        if self.verbose:
            logger.info(
                f"CE loss: {ce_loss.item():.6f}, obj: {obj_value:.6f}, "
                f"mean_prob: {mean_target_prob:.4f}"
            )
        
        return obj_value, metadata
