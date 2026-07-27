"""
Vanilla AutoDAN implementation with target phrase matching loss.
This is the original AutoDAN approach where the model is optimized to generate
a specific target phrase/response.
"""

import torch
import logging

from methods.autodan.base_autodan import BaseAutoDAN
from methods.autodan.autodan_utils import get_score_autodan

logger = logging.getLogger("vanilla_autodan")


class VanillaAutoDAN(BaseAutoDAN):
    """
    Vanilla AutoDAN that optimizes adversarial strings to maximize the probability
    of generating a target phrase. This is the original AutoDAN approach from the paper.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        target_phrase: str,
        num_steps: int = 100,
        batch_size: int = 256,
        num_elites: int = 13,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.01,
        num_points: int = 5,
        hga_interval: int = 5,
        system_prompt: str | None = None,
        prompt_format: str | None = None,
        allow_non_ascii: bool = False,
        verbose: bool = False,
        stagnation_patience: int = 20,
        initial_prompt_path: str | None = None,
        reference_prompts: list[str] | None = None,
        optim_str_mode: str = "suffix",
    ):
        """
        Initialize VanillaAutoDAN.
        
        Args:
            model: Target language model
            tokenizer: Model tokenizer
            target_phrase: Target phrase that the model should generate
            num_steps: Number of GA generations
            batch_size: Population size for GA
            num_elites: Number of top individuals to preserve each generation
            crossover_rate: Probability of crossover operation
            mutation_rate: Probability of mutation operation
            num_points: Number of crossover points
            hga_interval: Apply regular GA every N steps, HGA otherwise (0 to disable HGA)
            system_prompt: System prompt for chat template
            prompt_format: Format string for prompts
            allow_non_ascii: Allow non-ASCII characters in optimized strings
            verbose: Enable verbose logging
            stagnation_patience: Number of steps without improvement before early stopping
            initial_prompt_path: Path to file containing initial prompt
            reference_prompts: List of reference prompts for initialization
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            num_steps=num_steps,
            batch_size=batch_size,
            num_elites=num_elites,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            num_points=num_points,
            hga_interval=hga_interval,
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            allow_non_ascii=allow_non_ascii,
            verbose=verbose,
            stagnation_patience=stagnation_patience,
            initial_prompt_path=initial_prompt_path,
            reference_prompts=reference_prompts,
            optim_str_mode=optim_str_mode,
        )
        
        self.target_phrase = target_phrase
        logger.info(f"Initialized VanillaAutoDAN with target phrase: '{target_phrase}'")
    
    def _get_score_batch(
        self,
        input_context: str,
        instruction: str,
        test_controls: list[str],
    ) -> torch.Tensor:
        """
        Compute loss for a batch of control strings using target phrase matching.
        
        This implements the original AutoDAN loss: cross-entropy loss between
        the model's predicted tokens and the target phrase tokens. Lower loss
        means the model is more likely to generate the target phrase.
        
        Args:
            instruction: The query/instruction
            test_controls: List of adversarial control strings to evaluate
            
        Returns:
            Tensor of losses for each control string (lower is better)
        """
        # Build adversarial queries according to optim_str_mode
        adv_queries = [self._compose_adv_query(instruction, ctrl) for ctrl in test_controls]

        losses = get_score_autodan(
            tokenizer=self.tokenizer,
            model=self.generation_model,
            device=self.generation_model.device,
            input_context=input_context,
            instruction=None,  # unused when adv_queries provided
            target=self.target_phrase,
            test_controls=adv_queries,
            crit=self.crit,
            prompt_template=self.prompt_format,
            system_prompt=self.system_prompt,
        )
        
        return losses
    
    def run(self, input_context: str, query: str, expected_response: str, initial_generation: str | None = None):
        """
        Run VanillaAutoDAN to find adversarial suffixes that cause the model
        to generate the target phrase.
        
        Note: For VanillaAutoDAN, the target_phrase (set in __init__) is used
        instead of expected_response for loss computation. However, expected_response
        is still used for hallucination checking.
        
        Args:
            input_context: Context information for the query
            query: The original query
            expected_response: The expected/correct response (for hallucination checking)
            initial_generation: Optional initial generation string
            
        Returns:
            Dictionary containing attack results
        """
        if self.verbose:
            logger.info(
                f"Starting VanillaAutoDAN attack:\n"
                f"  Query: {query}\n"
                f"  Target phrase: {self.target_phrase}\n"
                f"  Expected response: {expected_response}\n"
                f"  Batch size: {self.batch_size}\n"
                f"  Num steps: {self.num_steps}"
            )
        
        # Call parent's run method
        result = super().run(input_context, query, expected_response, initial_generation)
        
        # Add target phrase to results
        result["target_phrase"] = self.target_phrase
        
        return result
