from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import logging

from methods.base_attack_method import BaseAttackMethod
from methods.autodan.autodan_utils import (
    autodan_sample_control,
    autodan_sample_control_hga,
    construct_initial_population,
)

logger = logging.getLogger("autodan")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BaseAutoDAN(BaseAttackMethod, ABC):
    """Base class for AutoDAN genetic algorithm-based adversarial attacks."""
    
    def __init__(
        self,
        model,
        tokenizer,
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
        Initialize AutoDAN attack method.
        
        Args:
            model: Target language model
            tokenizer: Model tokenizer
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
        super().__init__(model, tokenizer, system_prompt=system_prompt, prompt_format=prompt_format)
        
        # Core GA parameters
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_elites = num_elites
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_points = num_points
        self.hga_interval = hga_interval
        self.verbose = verbose
        self.stagnation_patience = stagnation_patience
        
        # Optimization string parameters
        supported_modes = ["suffix", "replace_query"]
        if optim_str_mode not in supported_modes:
            raise ValueError(f"Unsupported optim_str_mode: {optim_str_mode}. Supported: {supported_modes}")
        self.optim_str_mode = optim_str_mode
        self.optim_str_length = None
        
        # Non-ASCII filtering
        self.not_allowed_ids = None if allow_non_ascii else self._get_nonascii_toks(tokenizer, device=model.device)
        
        # Initial population setup
        self.initial_prompt_path = initial_prompt_path
        self.reference_prompts = reference_prompts or []
        
        # Load initial prompt if provided
        if initial_prompt_path is not None:
            with open(initial_prompt_path, 'r') as f:
                self.initial_prompt = ''.join(f.readlines())
        else:
            self.initial_prompt = "You are a helpful assistant. [REPLACE]"
        
        # Loss criterion (will be set by subclasses)
        self.crit = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # TODO: implement prefix cache for efficiency
        self.prefix_cache = None

    def run(self, input_context: str, query: str, expected_response: str, initial_generation: str | None = None):
        """
        Run AutoDAN genetic algorithm to find adversarial suffixes.
        
        Args:
            input_context: Context information for the query
            query: The original query
            expected_response: The expected/correct response
            initial_generation: Optional initial generation string
            
        Returns:
            Dictionary containing attack results
        """
        self.stop_flag = False
        
        # Store context and query for use in scoring
        self.input_context = input_context
        self.query = query
        
        # Check initial hallucination
        initial_hallucination_result = self.hallucination_check(
            input_context=input_context,
            query=query,
            expected_response=expected_response,
        )
        
        if initial_hallucination_result["hallucination_detected"] is True:
            logger.info("Initial query already causes hallucination. No need to run AutoDAN.")
            return {
                "original_query": query,
                "adversarial_query": query,
                "best_loss": None,
                "best_string": "",
                "steps": 0,
                "losses": [],
                "strings": [],
                "final_adversarial_response": initial_hallucination_result.get("model_response", ""),
                "final_hallucination": True,
                "final_hallucination_justification": initial_hallucination_result.get("justification", ""),
                "initial_response": initial_hallucination_result.get("model_response", ""),
                "initial_hallucination": True,
                "initial_hallucination_justification": initial_hallucination_result.get("justification", ""),
                "stagnation_triggered": False,
            }
        
        # Initialize population
        population = self._initialize_population(query)
        
        # Initialize word dictionary for HGA momentum
        word_dict = {}
        
        losses = []
        optim_strings = []
        best_loss = float('inf')
        best_string = population[0]
        current_best_string = best_string
        
        # Cache candidate losses across the run to avoid rescoring duplicates
        score_cache: dict[str, float] = {}
        score_dtype = torch.float32
        score_device = torch.device("cpu")
        
        # Stagnation tracking
        stagnation_counter = 0
        last_best_loss = float('inf')
        stagnation_triggered = False
        
        hallucination_result = {}
        
        # Main GA loop
        for step in tqdm(range(self.num_steps), desc="AutoDAN GA"):
            
            if self.stop_flag:
                logger.info("Early stopping triggered.")
                break
            
            # Evaluate fitness with memoization across iterations and dedup within the step
            unique_unseen_controls = []
            seen_unseen_controls = set()
            for control in population:
                if control in score_cache or control in seen_unseen_controls:
                    continue
                unique_unseen_controls.append(control)
                seen_unseen_controls.add(control)

            if len(unique_unseen_controls) > 0:
                unseen_scores = self._get_score_batch(
                    input_context=input_context,
                    instruction=query,
                    test_controls=unique_unseen_controls,
                )
                
                if unseen_scores.numel() != len(unique_unseen_controls):
                    raise RuntimeError(
                        "Mismatch between number of unseen controls and returned scores: "
                        f"{len(unique_unseen_controls)} vs {unseen_scores.numel()}"
                    )
                
                score_dtype = unseen_scores.dtype
                score_device = unseen_scores.device
                for control, score in zip(unique_unseen_controls, unseen_scores):
                    score_cache[control] = float(score.item())
            
            score_list = torch.tensor(
                [score_cache[control] for control in population],
                dtype=score_dtype,
                device=score_device,
            )
            
            # Track best individual
            current_best_idx = score_list.argmin().item()
            current_best_loss = score_list[current_best_idx].item()
            current_best_string = population[current_best_idx]
            
            losses.append(current_best_loss)
            optim_strings.append(current_best_string)
            
            # Update global best and only run hallucination check if global best improves
            is_new_global_best = current_best_loss < best_loss
            if is_new_global_best:
                best_loss = current_best_loss
                best_string = current_best_string
                
                # Check for hallucination only when the best loss decreases
                hallucination_result = self.hallucination_check(
                    input_context=input_context,
                    query=self._obtain_adversarial_query(query, current_best_string),
                    expected_response=expected_response,
                )
                
                if hallucination_result["hallucination_detected"] is True:
                    self.stop_flag = True
                    logger.info("Early stopping: hallucination detected.")
                    break
            
            # Stagnation check
            if self.stagnation_patience > 0:
                if current_best_loss < last_best_loss:
                    last_best_loss = current_best_loss
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if stagnation_counter >= self.stagnation_patience:
                    logger.info(
                        f"Early stopping due to stagnation: best loss unchanged for "
                        f"{self.stagnation_patience} iterations"
                    )
                    stagnation_triggered = True
                    self.stop_flag = True
                    break
            
            if self.verbose:
                hallucination_status = (
                    hallucination_result.get("hallucination_detected", False)
                    if is_new_global_best
                    else "skipped"
                )
                logger.info(
                    f"Step {step+1}/{self.num_steps} | "
                    f"Best Loss: {current_best_loss:.4f} | "
                    f"Best String: {current_best_string[:50]}... | "
                    f"Hallucination: {hallucination_status} | "
                    f"WordDict: {len(word_dict)}"
                )
            
            # Generate next generation using GA or HGA
            if self.hga_interval > 0 and step % self.hga_interval != 0:
                # Use HGA with word-level momentum
                if self.verbose:
                    logger.debug(f"Step {step + 1}: Using HGA")
                population, word_dict = autodan_sample_control_hga(
                    word_dict=word_dict,
                    control_suffixs=population,
                    score_list=score_list.cpu().numpy().tolist(),
                    num_elites=self.num_elites,
                    batch_size=self.batch_size,
                    crossover=self.crossover_rate,
                    mutation=self.mutation_rate,
                    API_key=None,
                    reference=self.reference_prompts if len(self.reference_prompts) > 0 else None,
                    if_api=False,
                )
            else:
                # Use regular GA
                if self.verbose and self.hga_interval > 0:
                    logger.debug(f"Step {step + 1}: Using regular GA")
                population = self._apply_crossover_and_mutation(
                    population,
                    score_list.cpu().numpy().tolist(),
                )
        
        # Use the overall best string found
        min_loss_index = losses.index(min(losses)) if len(losses) > 0 else 0
        optim_string = optim_strings[min_loss_index] if len(optim_strings) > 0 else ""
        best_loss = losses[min_loss_index] if len(losses) > 0 else None
        adversarial_query = self._obtain_adversarial_query(query, optim_string) if len(optim_strings) > 0 else query
        
        # Final hallucination check with best string
        if not hallucination_result or optim_string != current_best_string:
            hallucination_result = self.hallucination_check(
                input_context=input_context,
                query=adversarial_query,
                expected_response=expected_response,
            )

        return {
            "original_query": query,
            "adversarial_query": adversarial_query,
            "best_loss": best_loss,
            "best_string": optim_string,
            "steps": step + 1 if 'step' in locals() else 0,
            "losses": losses,
            "strings": optim_strings,
            "final_adversarial_response": hallucination_result.get("model_response", ""),
            "final_hallucination": hallucination_result.get("hallucination_detected", False),
            "final_hallucination_justification": hallucination_result.get("justification", ""),
            "initial_response": initial_hallucination_result.get("model_response", ""),
            "initial_hallucination": initial_hallucination_result.get("hallucination_detected", False),
            "initial_hallucination_justification": initial_hallucination_result.get("justification", ""),
            "stagnation_triggered": stagnation_triggered,
            "word_dict_size": len(word_dict),
        }
    
    def _initialize_population(self, query: str) -> list[str]:
        """
        Initialize the population of adversarial control strings.
        
        Args:
            query: The original query
            
        Returns:
            List of initial control strings
        """
        if len(self.reference_prompts) >= self.batch_size:
            population = self.reference_prompts[:self.batch_size]
            logger.info(f"Initialized population from {len(population)} reference prompts")
        else:
            if self.optim_str_mode == "suffix":
                population = construct_initial_population(
                    initial_prompt=self.initial_prompt,
                    batch_size=self.batch_size,
                    reference=self.reference_prompts if len(self.reference_prompts) > 0 else None,
                )
            else:
                population = construct_initial_population(
                    initial_prompt=query,
                    batch_size=self.batch_size,
                    reference=self.reference_prompts if len(self.reference_prompts) > 0 else None,
                    use_synonyms_on_initial=True,
                )
            logger.info(f"Initialized population with {len(population)} control strings")
        
        return population
    
    def _apply_crossover_and_mutation(
        self,
        population: list[str],
        score_list: list[float],
    ) -> list[str]:
        """
        Apply genetic algorithm operations to generate next generation.
        
        Args:
            population: Current population
            score_list: Fitness scores (losses) for each individual
            
        Returns:
            Next generation population
        """
        next_generation = autodan_sample_control(
            control_suffixs=population,
            score_list=score_list,
            num_elites=self.num_elites,
            batch_size=self.batch_size,
            crossover=self.crossover_rate,
            num_points=self.num_points,
            mutation=self.mutation_rate,
            API_key=None,
            reference=self.reference_prompts if len(self.reference_prompts) > 0 else None,
            if_softmax=True,
            if_api=False,
        )
        
        return next_generation
    
    @abstractmethod
    def _get_score_batch(
        self,
        input_context: str,
        instruction: str,
        test_controls: list[str],
    ) -> torch.Tensor:
        """
        Compute loss for a batch of control strings.
        This method should be implemented by subclasses to define
        the specific loss computation strategy.
        
        Args:
            instruction: The query/instruction
            test_controls: List of adversarial control strings to evaluate
            
        Returns:
            Tensor of losses for each control string
        """
        pass
    
    def _obtain_adversarial_query(self, query: str, optim_string: str) -> str:
        """Build the adversarial query according to optim_str_mode."""
        if self.optim_str_mode == "suffix":
            return (query + " " + optim_string).strip()
        return optim_string

    def _compose_adv_query(self, instruction: str, control_str: str) -> str:
        """Helper to build the query fed into scoring."""
        if self.optim_str_mode == "suffix":
            return f"{instruction} {control_str}".strip()
        return control_str
    
    @staticmethod
    def _get_nonascii_toks(tokenizer, device="cpu"):
        """Get token IDs for non-ASCII characters to filter them out."""
        def is_ascii(s):
            return s.isascii() and s.isprintable()

        nonascii_toks = []
        for i in range(tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                nonascii_toks.append(i)

        if tokenizer.bos_token_id is not None:
            nonascii_toks.append(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            nonascii_toks.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            nonascii_toks.append(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            nonascii_toks.append(tokenizer.unk_token_id)

        return torch.tensor(nonascii_toks, device=device)
