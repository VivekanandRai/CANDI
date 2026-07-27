from abc import ABC, abstractmethod
from tqdm import tqdm
import random

import torch
import logging
from torch import Tensor

# WordNet and the spell checker are loaded on first use, and the WordNet corpus
# is downloaded if missing - see methods/lazy_resources.py.
from methods.lazy_resources import (
    get_bert_fill_mask,
    get_bert_model,
    get_bert_tokenizer,
    get_sentence_transformer,
    get_spell_checker,
    get_wordnet,
)

from methods.base_attack_method import BaseAttackMethod

logger = logging.getLogger("gcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


PUNCTUATION_TOKENS = [".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'"]
SEMANTIC_EQUIVALENCE_CHECK_MODEL = "openai/gpt-5-nano"

def get_synonyms_wordnet(word):
    """Get all synonyms for a word using WordNet."""
    synonyms = set()
    for syn in get_wordnet().synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

class BaseGCG(BaseAttackMethod, ABC):
    def __init__(
        self,
        model,
        tokenizer,
        top_k: int = 10,
        num_steps: int = 100,
        search_width: int = 5,
        optim_str_mode: str = "suffix",
        batch_size: int = 1,
        optim_str_constraint: str | None = None,
        optim_str_length: int | None = None,
        system_prompt: str | None = None,
        prompt_format: str | None = None,
        buffer_size: int = 0,
        allow_non_ascii: bool = False,
        n_replace: int = 1,
        verbose: bool = False,
        stagnation_patience: int = 20,
        perplexity_threshold: float | None = None,
        cosine_similarity_threshold: float = 0.99,
        filter_candidates_by_llm_semantic_equivalence: bool = False,
    ):
        
        
        super().__init__(model, tokenizer, system_prompt=system_prompt, prompt_format=prompt_format)
        self.embedding_layer = model.get_input_embeddings()
        self.top_k = top_k
        self.num_steps = num_steps
        self.search_width = search_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.verbose = verbose
        
        supported_optim_str_modes = ["suffix", "replace_query"]
        supported_optim_str_constraints = [None, "synonym_replacement", "synonym_replacement_most_important", "synonym_replacement_most_important_ppl_filter", "punctuation_only", "bert_synonym"]
        
        if optim_str_mode not in supported_optim_str_modes:
            raise ValueError(f"Unsupported optim_str_mode: {optim_str_mode}. Supported modes are: {supported_optim_str_modes}")
        
        if optim_str_constraint not in supported_optim_str_constraints:
            raise ValueError(f"Unsupported optim_str_constraint: {optim_str_constraint}. Supported constraints are: {supported_optim_str_constraints}")
        
        if optim_str_mode == "replace_query" and optim_str_length is not None and optim_str_constraint != "punctuation_only":
            raise ValueError("optim_str_length is not supported in replace_query mode (except for punctuation_only)")
        
        if optim_str_mode == "suffix" and optim_str_length is None:
            raise ValueError("optim_str_length must be specified in suffix mode")
        
        if optim_str_constraint == "punctuation_only" and optim_str_mode != "replace_query":
            raise ValueError("punctuation_only constraint requires replace_query mode")
        
        self.optim_str_mode = optim_str_mode
        self.optim_str_constraint = optim_str_constraint
        self.optim_str_length = optim_str_length
        self.not_allowed_ids = None if allow_non_ascii else self._get_nonascii_toks(tokenizer, device=model.device)
        self.n_replace = n_replace
        self.stagnation_patience = stagnation_patience
        self.perplexity_threshold = perplexity_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.filter_candidates_by_llm_semantic_equivalence = filter_candidates_by_llm_semantic_equivalence
        
        # Initialize punctuation-specific attributes
        self.punctuation_positions = None
        self.punctuation_token_ids = None
        if optim_str_constraint == "punctuation_only":
            self.punctuation_token_ids = self._get_punctuation_tokens(tokenizer)

        # TODO: implement prefix cache
        self.prefix_cache = None
        
        self.punctuation_token_ids_2 = set([self.tokenizer(punct, add_special_tokens=False)["input_ids"][0] for punct in PUNCTUATION_TOKENS])
        
        if optim_str_constraint == "bert_synonym":
            # Masked-LM synonym proposal. These are process-wide cached, so the
            # download cost is paid once even across several attack instances.
            self.bert_tokenizer = get_bert_tokenizer()
            self.bert_model = get_bert_model()
            self.bert_fill_mask = get_bert_fill_mask()
            self.sentence_transformer = get_sentence_transformer()


    def run(self, input_context: str, query: str, expected_response: str, initial_generation: str | None = None):
        
        self.stop_flag = False
        
        before_str, after_str = self._split_context_prompt_into_parts(input_context, query)
        
        if initial_generation is not None:
            # adjust after_str to account for initial generation
            after_str = initial_generation + after_str
        
        before_ids = self.tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.model.device, torch.int64)
        after_ids = self.tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device, torch.int64)
        
        
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds,  = [embedding_layer(ids) for ids in (before_ids, after_ids)]

        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        
        init_optim_ids = self._initialize_optim_ids(query)
        
        buffer = self.init_buffer(init_optim_ids)
        
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        hallucination_result = {}
        
        initial_hallucination_result = self.hallucination_check(
            input_context=input_context,
            query=query,
            expected_response=expected_response,
        )
        step = 0
        if initial_hallucination_result["hallucination_detected"] is True:
            logger.info("Initial query already causes hallucination. No need to run GCG.")
            self.stop_flag = True
            
            hallucination_result = initial_hallucination_result
        
        # Stagnation tracking
        stagnation_counter = 0
        last_best_loss = float('inf')
        stagnation_triggered = False
        
        for step in tqdm(range(self.num_steps)):
            # Compute the token gradient
            
            if self.stop_flag:
                logger.info("Early stopping triggered.")
                break
            
            optim_ids_onehot_grad = self._compute_gradients(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = self._sample_tokens_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                )
                
            new_search_width = sampled_ids.shape[0]  # Use actual number of unique sequences
            
            loss = self._compute_loss_batch(sampled_ids, new_search_width)
            
            current_loss = loss.min().item()
            optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
            
            # Update the buffer based on the loss
            losses.append(current_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            
            # TODO: add the hallucination checker for early stopping
            hallucination_result = self.hallucination_check(
                input_context=input_context,
                query=self._obtain_adversarial_query(query, optim_str),
                expected_response=expected_response,
            )
            
            if hallucination_result["hallucination_detected"] is True:
                self.stop_flag = True
                logger.info("Early stopping: hallucination detected.")
            
            # Stagnation check: if best loss hasn't improved for stagnation_patience iterations
            if self.stagnation_patience > 0:
                current_best_loss = buffer.get_lowest_loss()
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
            
            if self.verbose:
                logger.info(f"Step {step+1}/{self.num_steps} | Current Loss: {current_loss} | Optimized String: {optim_str} | Model Response: {hallucination_result.get('model_response', '')}")
            
            
        min_loss_index = losses.index(min(losses)) if len(losses) > 0 else 0
        optim_string = optim_strings[min_loss_index] if len(optim_strings) > 0 else ""
        best_loss = losses[min_loss_index] if len(losses) > 0 else None
        adversarial_query = self._obtain_adversarial_query(query, optim_string) if len(optim_strings) > 0 else query

        return {
            "original_query": query,
            "adversarial_query": adversarial_query,
            "best_loss": best_loss,
            "best_string": optim_string,
            "steps": step + 1,
            "losses": losses,
            "strings": optim_strings,
            "final_adversarial_response": hallucination_result.get("model_response", ""),
            "final_hallucination": hallucination_result.get("hallucination_detected", False),
            "final_hallucination_justification": hallucination_result.get("justification", ""),
            "initial_response": initial_hallucination_result.get("model_response", ""),
            "initial_hallucination": initial_hallucination_result.get("hallucination_detected", False),
            "initial_hallucination_justification": initial_hallucination_result.get("justification", ""),
            "stagnation_triggered": stagnation_triggered,
        }
        
    def determine_if_token_is_full_word(self, tokens, position):
        token_id = tokens["input_ids"][0][position]
        word_ids = tokens.word_ids()
        token_word_id = word_ids[position]
        
        # Check if the token is the only token for its word ID
        if token_word_id is None:
            return False  # Special token or padding
        
        # Count how many tokens correspond to this word ID
        
        if word_ids[position - 1] == token_word_id or (position + 1 < len(word_ids) and word_ids[position + 1] == token_word_id):
            return False 
        
        token_text = self.tokenizer.decode(token_id).strip()
        
        if token_text in get_spell_checker():
            return True
        
        else:
            return False
        
    def _get_simple_embedding_synonyms(
        self,
        replacement_token,
        topk: int = 20, 
    ):
        token_embedding = self.model.embed_tokens(replacement_token)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        similarities = cos(
            token_embedding.unsqueeze(0),
            self.model.embed_tokens.weight
        )

        top_k_indices = torch.topk(similarities, topk).indices
        
        return top_k_indices
    
    def _get_nltk_synonyms_full_words(
        self,
        token_sequence,
        position,
        # replacement_token,
        topk: int = 20, 
    ):
        """
        Finds the top_k best synonym tokens for a word at a specific position in a token sequence.
        Preserves leading spaces in tokens to maintain proper formatting.
        Returns a list of token IDs of the top_k synonyms.
        """
        
        # --- Get the target token and check for leading space ---
        target_token_id = token_sequence[position]
        target_token_text = self.tokenizer.decode([target_token_id])
        
        # Check if the token has a leading space
        has_leading_space = target_token_text.startswith(' ')
        target_word = target_token_text.strip()
        
        if self.verbose:
            logger.info(f"Word to be replaced: {target_word} (has_leading_space={has_leading_space})")
        
        # --- Decode the original sentence for context ---
        original_sentence = self.tokenizer.decode(token_sequence, skip_special_tokens=True)
        if self.verbose:
            logger.info(f"Original sentence: {original_sentence}")
        
        # --- Step 1: Get Raw Candidates from WordNet ---
        raw_synonyms = set()
        for syn in get_wordnet().synsets(target_word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ')
                if candidate.lower() != target_word.lower():
                    raw_synonyms.add(candidate)
        
        if not raw_synonyms:
            if self.verbose:
                logger.info(f"No synonyms found for '{target_word}', returning original token.")
            return [target_token_id]

        # --- Step 2: Score Candidates with LLM (Contextual) ---
        candidate_scores = []
        
        def score_sentence(sentence):
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.generation_model(**inputs, labels=inputs.input_ids)
            return outputs.loss.item()

        if self.verbose:
            logger.info(f"Ranking {len(raw_synonyms)} candidates...: {raw_synonyms}")
        
        for syn in raw_synonyms:
            test_sentence = original_sentence.replace(target_word, syn)
            
            if test_sentence != original_sentence:
                loss = score_sentence(test_sentence)
                candidate_scores.append((syn, loss))

        # --- Step 3: Sort by Lowest Loss and Get Top-K Synonyms ---
        candidate_scores.sort(key=lambda x: x[1])
        
        if not candidate_scores:
            if self.verbose:
                logger.info(f"No valid candidates scored, returning original token.")
            return [target_token_id]
        
        # Get top_k synonyms
        top_synonyms = candidate_scores[:topk]
        
        
        # --- Step 4: Tokenize the top_k synonyms and return their token IDs ---
        # Preserve leading space and only keep synonyms that tokenize to exactly 1 token
        top_synonym_token_ids = []
        for syn, _ in top_synonyms:
            # Add leading space if the original token had one
            syn_text = ' ' + syn if has_leading_space else syn
            syn_tokens = self.tokenizer.encode(syn_text, add_special_tokens=False)
            
            # Only include synonyms that are single tokens
            if len(syn_tokens) == 1:
                top_synonym_token_ids.append(syn_tokens[0])
        
        # Fallback: if no single-token synonyms found, return original token
        if len(top_synonym_token_ids) == 0:
            if self.verbose:
                logger.info(f"No single-token synonyms found for '{target_word}', returning original token.")
            return [target_token_id]
        
        return top_synonym_token_ids
        
    
    def _get_punctuation_tokens(self, tokenizer):
        """Get token IDs for defined punctuation characters."""
        punctuation_chars = [' ', ',', '.', ';', ':', '\n', '\t', '!', '?', '-', '(', ')']
        punctuation_ids = []
        
        for char in punctuation_chars:
            token_ids = tokenizer(char, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            # Handle cases where a character might tokenize to multiple tokens (take first)
            if len(token_ids) > 0:
                punctuation_ids.append(token_ids[0].item())
        
        # Remove duplicates and return as tensor
        punctuation_ids = list(set(punctuation_ids))
        return torch.tensor(punctuation_ids, device=self.model.device)
    
    def _split_and_tokenize_query(self, query: str):
        """Split query into words, tokenize each, and join with space tokens.
        
        Returns:
            token_ids: (1, seq_len) tensor of token IDs
            punctuation_positions: list of indices where space tokens were inserted
        """
        import re
        
        # Define punctuation pattern for splitting
        punctuation_pattern = r'[ ,\.;:\n\t!?\-\(\)]'
        
        # Split the query while preserving the structure
        words = [word for word in re.split(punctuation_pattern, query) if word]
        
        if len(words) == 0:
            raise ValueError("Query resulted in no words after splitting")
        
        # Tokenize each word separately
        word_token_lists = []
        for word in words:
            tokens = self.tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            word_token_lists.append(tokens)
        
        # Get the space token ID
        space_token_id = self.tokenizer(' ', add_special_tokens=False, return_tensors="pt")["input_ids"][0, 0]
        
        # Concatenate with space tokens between words
        all_tokens = []
        punctuation_positions = []
        
        for i, word_tokens in enumerate(word_token_lists):
            all_tokens.extend(word_tokens.tolist())
            
            # Add space token between words (but not after the last word)
            if i < len(word_token_lists) - 1:
                punctuation_positions.append(len(all_tokens))
                all_tokens.append(space_token_id.item())
        
        token_ids = torch.tensor([all_tokens], device=self.model.device)
        
        return token_ids, punctuation_positions
    
    def _deduplicate_sequences(self, sampled_ids):
        """
        Remove duplicate sequences from sampled_ids, returning only unique sequences.
        This reduces the batch size to the number of unique sequences.
        
        Args:
            sampled_ids: Tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (num_unique, seq_len) with only unique sequences
        """
        # Find unique sequences
        unique_seqs = []
        seen = set()
        
        for seq in sampled_ids:
            seq_tuple = tuple(seq.tolist())
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique_seqs.append(seq)
        
        if len(unique_seqs) == 0:
            # Shouldn't happen, but safeguard
            return sampled_ids[:1]  # Return at least one sequence
        
        # Stack and return unique sequences
        return torch.stack(unique_seqs)
        
    def _sample_tokens_from_grad(self, optim_ids, grad, ):
        
        n_optim_tokens = len(optim_ids)
        original_ids = optim_ids.repeat(self.search_width, 1)
        if self.not_allowed_ids is not None:
            grad[:, self.not_allowed_ids.to(grad.device)] = float("inf")
        
        if self.optim_str_constraint is None:

            topk_ids = (-grad).topk(self.top_k, dim=1).indices

            sampled_ids_pos = torch.argsort(torch.rand((self.search_width, n_optim_tokens), device=grad.device))[..., :self.n_replace]
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                torch.randint(0, self.top_k, (self.search_width, self.n_replace, 1), device=grad.device),
            ).squeeze(2)

            new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

            return new_ids
        
        elif self.optim_str_constraint == "synonym_replacement_deprecated":
            # do a constrained sampling based on synonyms
           
            # identify an index to replace:
            
            replacement_position = random.randint(0, n_optim_tokens - 1)
            
            replacement_token = optim_ids[replacement_position]
            
            
            top_k_indices = self._get_simple_embedding_synonyms(
                replacement_token,
                topk=self.top_k,
            )

            # only consider these topk indices for replacement - make the others have inf grad
            mask = torch.ones_like(grad[replacement_position], dtype=torch.bool)
            mask[top_k_indices] = False
            
            grad[replacement_position][mask] = float("inf")
        
            topk_ids = (-grad).topk(self.top_k, dim=1).indices

            sampled_ids_pos = torch.full((self.search_width, 1), replacement_position, device=grad.device, dtype=torch.long)
            
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                torch.randint(0, self.top_k, (self.search_width, self.n_replace, 1), device=grad.device),
            ).squeeze(2)

            new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

            return new_ids
        
        elif self.optim_str_constraint == "synonym_replacement":
            # Identify valid positions at token level (no re-tokenization)
            valid_positions = []
            for pos in range(n_optim_tokens):
                token_id = optim_ids[pos]
                token_text = self.tokenizer.decode([token_id])
                
                # Strip any leading/trailing spaces to analyze the core token
                stripped_text = token_text.strip()
                
                # Only consider tokens that:
                # 1. Are alphabetic words (not punctuation/numbers)
                # 2. Are at least 2 characters long
                # 3. Are valid English words (in get_spell_checker() checker dictionary)
                if (len(stripped_text) >= 2 and 
                    stripped_text.isalpha() and 
                    stripped_text.lower() in get_spell_checker()):
                    valid_positions.append(pos)
            
            if len(valid_positions) == 0:
                # No valid positions found, return original sequence
                return optim_ids.unsqueeze(0).repeat(self.search_width, 1)
            
            replacement_position = random.choice(valid_positions)
            
            top_k_indices = self._get_nltk_synonyms_full_words(
                optim_ids,
                replacement_position,
                topk=self.top_k,
            )

            # only consider these topk indices for replacement - make the others have inf grad
            mask = torch.ones_like(grad[replacement_position], dtype=torch.bool)
            mask[top_k_indices] = False
            
            grad[replacement_position][mask] = float("inf")
        
            topk_ids = (-grad).topk(self.top_k, dim=1).indices

            sampled_ids_pos = torch.full((self.search_width, 1), replacement_position, device=grad.device, dtype=torch.long)
            
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                torch.randint(0, self.top_k, (self.search_width, self.n_replace, 1), device=grad.device),
            ).squeeze(2)

            new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

            return new_ids
        
        elif self.optim_str_constraint == "synonym_replacement_most_important":
            # Gradient-based synonym replacement: find tokens with highest gradients
            
            # Extract gradients for the actual tokens in the sequence
            # grad shape: (n_optim_tokens, vocab_size), optim_ids shape: (n_optim_tokens,)
            token_grads = grad[torch.arange(n_optim_tokens, device=grad.device), optim_ids]
            
            # Find top-k positions with highest gradients (most important to change)
            num_candidate_positions = min(self.top_k, n_optim_tokens)
            top_positions = torch.topk(token_grads, num_candidate_positions).indices
            
            # Distribute search_width across candidate positions
            candidates_per_position = max(1, self.search_width // num_candidate_positions)
            total_candidates = candidates_per_position * num_candidate_positions
            
            all_new_ids = []
            
            for pos_idx, replacement_position in enumerate(top_positions.tolist()):
                replacement_token = optim_ids[replacement_position]
                
                top_k_indices = self._get_simple_embedding_synonyms(
                    replacement_token,
                    topk=self.top_k + 1,  # +1 to account for potential self-token
                )
                
                # Prevent self-substitution: remove the original token from candidates
                top_k_indices = top_k_indices[top_k_indices != replacement_token.item()]
                top_k_indices = top_k_indices[:self.top_k]  # Trim back to top_k
                
                if len(top_k_indices) == 0:
                    continue
                
                # Create candidates for this position
                pos_original_ids = optim_ids.unsqueeze(0).repeat(candidates_per_position, 1)
                
                # Sample from the filtered synonyms
                sampled_indices = torch.randint(0, len(top_k_indices), (candidates_per_position,), device=grad.device)
                sampled_ids_val = top_k_indices[sampled_indices]
                
                # Replace at this position
                pos_original_ids[:, replacement_position] = sampled_ids_val
                all_new_ids.append(pos_original_ids)
            
            if len(all_new_ids) == 0:
                # Fallback: return original ids repeated
                return optim_ids.unsqueeze(0).repeat(self.search_width, 1)
            
            new_ids = torch.cat(all_new_ids, dim=0)
            
            # Trim or pad to match search_width
            if new_ids.shape[0] > self.search_width:
                new_ids = new_ids[:self.search_width]
            elif new_ids.shape[0] < self.search_width:
                # Pad by repeating some candidates
                repeat_count = (self.search_width - new_ids.shape[0])
                padding = new_ids[:repeat_count]
                new_ids = torch.cat([new_ids, padding], dim=0)

            if self.verbose:
                logger.info(f"Modified candidates in batch ({new_ids.shape[0]} candidates):")
                for i, candidate_ids in enumerate(new_ids):
                    decoded_sentence = self.tokenizer.decode(candidate_ids.tolist())
                    logger.info(f"  [{i}]: {decoded_sentence}")

            return new_ids
        
        elif self.optim_str_constraint == "synonym_replacement_most_important_ppl_filter":
            # Gradient-based synonym replacement with perplexity filtering
            
            # Extract gradients for the actual tokens in the sequence
            token_grads = grad[torch.arange(n_optim_tokens, device=grad.device), optim_ids]
            
            # Find top-k positions with highest gradients (most important to change)
            num_candidate_positions = min(self.top_k, n_optim_tokens)
            top_positions = torch.topk(token_grads, num_candidate_positions).indices
            
            # Distribute search_width across candidate positions
            candidates_per_position = max(1, self.search_width // num_candidate_positions)
            total_candidates = candidates_per_position * num_candidate_positions
            
            all_new_ids = []
            
            for pos_idx, replacement_position in enumerate(top_positions.tolist()):
                replacement_token = optim_ids[replacement_position]
                
                top_k_indices = self._get_simple_embedding_synonyms(
                    replacement_token,
                    topk=self.top_k + 1,  # +1 to account for potential self-token
                )
                
                # Prevent self-substitution: remove the original token from candidates
                top_k_indices = top_k_indices[top_k_indices != replacement_token.item()]
                top_k_indices = top_k_indices[:self.top_k]  # Trim back to top_k
                
                if len(top_k_indices) == 0:
                    continue
                
                # Create candidates for this position
                pos_original_ids = optim_ids.unsqueeze(0).repeat(candidates_per_position, 1)
                
                # Sample from the filtered synonyms
                sampled_indices = torch.randint(0, len(top_k_indices), (candidates_per_position,), device=grad.device)
                sampled_ids_val = top_k_indices[sampled_indices]
                
                # Replace at this position
                pos_original_ids[:, replacement_position] = sampled_ids_val
                all_new_ids.append(pos_original_ids)
            
            if len(all_new_ids) == 0:
                # Fallback: return original ids repeated
                return optim_ids.unsqueeze(0).repeat(self.search_width, 1)
            
            new_ids = torch.cat(all_new_ids, dim=0)
            
            # Trim or pad to match search_width
            if new_ids.shape[0] > self.search_width:
                new_ids = new_ids[:self.search_width]
            elif new_ids.shape[0] < self.search_width:
                # Pad by repeating some candidates
                repeat_count = (self.search_width - new_ids.shape[0])
                padding = new_ids[:repeat_count]
                new_ids = torch.cat([new_ids, padding], dim=0)
            
            # Apply perplexity filtering
            if self.perplexity_threshold is not None:
                new_ids = self._filter_by_perplexity(new_ids, optim_ids.unsqueeze(0).repeat(self.search_width, 1))
            
            if self.verbose:
                logger.info(f"Modified candidates in batch ({new_ids.shape[0]} candidates):")
                for i, candidate_ids in enumerate(new_ids):
                    decoded_sentence = self.tokenizer.decode(candidate_ids.tolist())
                    logger.info(f"  [{i}]: {decoded_sentence}")
            
            # TODO: Implement soft scoring alternative that combines attack_loss + λ × perplexity
            # instead of hard filtering

            return new_ids
        
        elif self.optim_str_constraint == "punctuation_only":
            # Punctuation-only optimization: only modify tokens at punctuation positions
            
            if self.punctuation_positions is None or len(self.punctuation_positions) == 0:
                logger.warning("No punctuation positions found. Returning original ids.")
                return optim_ids.unsqueeze(0).repeat(self.search_width, 1)
            
            # Mask gradients: only keep gradients at punctuation positions
            masked_grad = grad.clone()
            mask = torch.ones(n_optim_tokens, dtype=torch.bool, device=grad.device)
            mask[self.punctuation_positions] = False
            masked_grad[mask] = float("inf")  # Infinite gradient for non-punctuation positions
            
            # Mask candidate tokens: only allow punctuation tokens
            vocab_mask = torch.ones(masked_grad.shape[1], dtype=torch.bool, device=grad.device)
            vocab_mask[self.punctuation_token_ids] = False
            masked_grad[:, vocab_mask] = float("inf")  # Infinite gradient for non-punctuation tokens
            
            # Get top-k punctuation tokens at punctuation positions
            topk_ids = (-masked_grad).topk(self.top_k, dim=1).indices
            
            # Sample positions to modify (only from punctuation positions)
            num_positions_to_modify = min(self.n_replace, len(self.punctuation_positions))
            
            all_new_ids = []
            for _ in range(self.search_width):
                new_candidate = optim_ids.clone()
                
                # Randomly select which punctuation positions to modify
                positions_to_modify = random.sample(self.punctuation_positions, num_positions_to_modify)
                
                # For each selected position, sample a punctuation token
                for pos in positions_to_modify:
                    # Get valid punctuation candidates for this position
                    valid_candidates = topk_ids[pos]
                    # Filter to only include punctuation tokens
                    valid_candidates = valid_candidates[torch.isin(valid_candidates, self.punctuation_token_ids)]
                    
                    if len(valid_candidates) > 0:
                        # Randomly select one
                        sampled_idx = torch.randint(0, len(valid_candidates), (1,), device=grad.device)
                        new_candidate[pos] = valid_candidates[sampled_idx]
                
                all_new_ids.append(new_candidate)
            
            new_ids = torch.stack(all_new_ids, dim=0)
            
            if self.verbose:
                logger.info(f"Punctuation-only optimization at positions {self.punctuation_positions}:")
                for i, candidate_ids in enumerate(new_ids):
                    decoded_sentence = self.tokenizer.decode(candidate_ids.tolist())
                    logger.info(f"  [{i}]: {decoded_sentence}")
            
            return new_ids
        
        elif self.optim_str_constraint == "bert_synonym":
            # Identify valid positions at token level (no re-tokenization)
            
    
            valid_positions = []
            for pos in range(n_optim_tokens):
                if self._determine_if_token_is_full_word(optim_ids, pos):
                    valid_positions.append(pos)
            
            if len(valid_positions) == 0:
                # No valid positions found, return original sequence
                return optim_ids.unsqueeze(0).repeat(self.search_width, 1)
            
            replacement_position = random.choice(valid_positions)
            original_decoded_token = self.tokenizer.decode([optim_ids[replacement_position]])
            
            add_leading_space = False
            
            if original_decoded_token.startswith(" "):
                add_leading_space = True
            
            original_word = self.tokenizer.decode(optim_ids[replacement_position]).strip()
            original_sentence = self.tokenizer.decode(optim_ids, skip_special_tokens=True)
            
            prefix_str = self.tokenizer.decode(optim_ids[:replacement_position])
            suffix_str = self.tokenizer.decode(optim_ids[replacement_position+1:])
            
            masked_sentence = prefix_str + " " + self.bert_tokenizer.mask_token + " " + suffix_str
    
            predictions = self.bert_fill_mask(masked_sentence, top_k=self.top_k)
                
            
            contextual_synonyms = [pred['token_str'].strip() for pred in predictions if pred['score'] >= 0.01 and pred['token_str'].strip().lower() in get_spell_checker()]
            
            
            synonym_tok_ids = []
            
            
            wordnet_synonyms = get_synonyms_wordnet(original_word)
            embedding_synonym_tokens = self._get_simple_embedding_synonyms(optim_ids[replacement_position], topk=self.top_k)
            
            embedding_synonyms = [self.tokenizer.decode([syn_id]) for syn_id in embedding_synonym_tokens]
            all_synonyms = set(contextual_synonyms + wordnet_synonyms + embedding_synonyms)
            
            for synonym_word in all_synonyms:
                
                word = synonym_word
                
                if word not in get_spell_checker():
                    continue
                
                modified_sentence = masked_sentence.replace(self.bert_tokenizer.mask_token, word)
                    
                # cosine_sim = self._get_bert_cosine_similarity(original_sentence, modified_sentence)
                sentence_transformer_sim = self._get_sentence_transformer_cosine_similarity(original_sentence, modified_sentence)
                
                # if self.verbose:
                #     logger.info(f"Evaluating sentence: '{modified_sentence}' | Cosine Similarity: {cosine_sim:.4f} | Sentence Transformer Similarity: {sentence_transformer_sim:.4f} | Original Word: '{original_word}' | Synonym: '{word}'")
                
                if sentence_transformer_sim < self.cosine_similarity_threshold: 
                    continue
                
                if add_leading_space:
                    word = " " + word
                
                synonym_tok_id = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                if len(synonym_tok_id) == 1:
                    synonym_tok_ids.append(synonym_tok_id[0])
                    
            top_k_indices = synonym_tok_ids
            
            # if no synonyms found, return original ids
            if len(top_k_indices) == 0:
                return optim_ids.unsqueeze(0)
            
            num_synonyms = len(top_k_indices)
            topk_current = min(self.top_k, num_synonyms)

            # only consider these topk indices for replacement - make the others have inf grad
            mask = torch.ones_like(grad[replacement_position], dtype=torch.bool)
            mask[top_k_indices] = False
            
            grad[replacement_position][mask] = float("inf")
        
            topk_ids = (-grad).topk(topk_current, dim=1).indices

            sampled_ids_pos = torch.full((self.search_width, 1), replacement_position, device=grad.device, dtype=torch.long)
            
            sampled_ids_val = torch.gather(
                topk_ids[sampled_ids_pos],
                2,
                torch.randint(0, topk_current, (self.search_width, self.n_replace, 1), device=grad.device),
            ).squeeze(2)

            new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
            
            
            # Debugging output

            new_ids = self._deduplicate_sequences(new_ids)
            
            if self.verbose:
                logger.info(f"Modified candidates in batch ({new_ids.shape[0]} candidates) with BERT synonyms:")
                    
            
            if self.filter_candidates_by_llm_semantic_equivalence:
                filtered_new_ids = []
                
                for i, candidate_ids in enumerate(new_ids):
                    decoded_sentence = self.tokenizer.decode(candidate_ids.tolist())
                    
                    # is_equivalent = self._semantic_equivalence_via_llm_fast(original_sentence, decoded_sentence, SEMANTIC_EQUIVALENCE_CHECK_MODEL)
                    is_equivalent = True
                    
                    if is_equivalent is None:
                        is_equivalent = True  # Default to True if LLM fails to provide a clear answer
                    
                    if self.verbose:
                        logger.info(f"  [{i}]: {decoded_sentence} | Semantic Equivalence: {is_equivalent}")
                    
                    if is_equivalent:
                        filtered_new_ids.append(candidate_ids)
                
                if len(filtered_new_ids) == 0:
                    logger.warning("All candidates were filtered out by semantic equivalence check. Returning original ids.")
                    return optim_ids.unsqueeze(0)  # Return 2D tensor with shape (1, seq_len)
                
                return torch.stack(filtered_new_ids)
                    
            else:
                return new_ids
                    
           
    def _determine_if_token_is_full_word(self, tokens, position):
        token_id = tokens[position]
        next_token_id = tokens[position + 1] if position + 1 < len(tokens) else None
        
        token_text = self.tokenizer.decode(token_id).strip()
        
        if next_token_id is not None:
            next_token_text = self.tokenizer.decode(next_token_id).strip()
          
        try:  
            next_token_is_punctuation = next_token_id in self.punctuation_token_ids_2 if next_token_id is not None else True
        except Exception as exc:
            logger.error(f"Error checking if next token is punctuation: {exc}")
            next_token_is_punctuation = True  # Default to True to avoid false negatives
        
        if (token_text in get_spell_checker()) and (next_token_text in get_spell_checker() or next_token_is_punctuation):
            return True
        
        else:
            return False
        
    def _compute_perplexity_batch(self, candidate_ids: Tensor) -> Tensor:
        """
        Compute perplexity for each candidate sequence (optim_ids only for speed).
        
        Args:
            candidate_ids: (batch_size, seq_len) - the optim_ids candidates
        
        Returns:
            perplexities: (batch_size,) - perplexity for each candidate
        """
        with torch.no_grad():
            # Get logits for the candidate sequences
            inputs_embeds = self.embedding_layer(candidate_ids)
            outputs = self.generation_model(inputs_embeds=inputs_embeds)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = candidate_ids[:, 1:].contiguous()
            
            # Compute cross-entropy loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Reshape for loss computation
            batch_size, seq_len = shift_labels.shape
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_size, seq_len)
            
            # Mean loss per sequence, then exponentiate for perplexity
            perplexities = torch.exp(loss.mean(dim=1))
            
        return perplexities
    
    def _filter_by_perplexity(self, new_ids: Tensor, original_ids: Tensor) -> Tensor:
        """
        Filter candidates by perplexity threshold.
        
        Args:
            new_ids: (search_width, seq_len) - candidate sequences
            original_ids: (search_width, seq_len) - fallback if all rejected
        
        Returns:
            filtered_ids: candidates that pass the threshold, 
                          or original_ids[0] repeated if all rejected
        """
        assert self.perplexity_threshold is not None, "perplexity_threshold must be set to use this method"
        
        perplexities = self._compute_perplexity_batch(new_ids)
        
        # Find candidates below threshold
        valid_mask = perplexities < self.perplexity_threshold
        
        if valid_mask.sum() == 0:
            # All candidates rejected, fall back to original (no change)
            logger.debug(f"All candidates rejected by perplexity filter (min ppl: {perplexities.min().item():.2f}, threshold: {self.perplexity_threshold})")
            return original_ids
        
        # Keep only valid candidates
        valid_ids = new_ids[valid_mask]
        
        # If fewer valid candidates than search_width, repeat to fill
        if valid_ids.shape[0] < self.search_width:
            repeat_times = (self.search_width + valid_ids.shape[0] - 1) // valid_ids.shape[0]
            valid_ids = valid_ids.repeat(repeat_times, 1)[:self.search_width]
        
        return valid_ids
        
    def _split_context_prompt_into_parts(self, context: str, query: str):
    
        if self.optim_str_mode == "replace_query":
            placeholder = "{optim_str}"
            messages = [
				{"role": "system", "content": self.system_prompt},
				{"role": "user", "content": self.prompt_format.format(context=context, query=placeholder)}
			]
            full_prompt = self.tokenizer.apply_chat_template(
				messages, tokenize=False, add_generation_prompt=True
			)
            before_str, after_str = full_prompt.split(placeholder)
            
            return before_str, after_str
        
        else:
			
            full_prompt = self.apply_chat_template_to_context_query(context, query)
            placeholder = "{optim_str}"
            if placeholder not in full_prompt:
                full_prompt = full_prompt + placeholder
            try:
                before_str, after_str = full_prompt.split(placeholder)
            except ValueError as exc:  
                raise RuntimeError("Unable to split prompt on {optim_str} placeholder") from exc
            return before_str, after_str
    
    def _initialize_optim_ids(self, query: str = ""):
        if self.optim_str_mode == "suffix":
            """Start from a simple token like 'x'; subclasses can extend later."""
            init_char = "!"
            
            init_id = self.tokenizer(init_char, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
            
            ids = init_id.repeat(1, self.optim_str_length)
            
            return ids
        
        elif self.optim_str_mode == "replace_query":
            if self.optim_str_constraint == "punctuation_only":
                # Use special tokenization that separates words with spaces
                ids, punctuation_positions = self._split_and_tokenize_query(query)
                self.punctuation_positions = punctuation_positions
                logger.info(f"Initialized punctuation_only mode with {len(punctuation_positions)} punctuation positions: {punctuation_positions}")
                return ids
            else:
                optim_str = query
                ids = self.tokenizer(optim_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
                return ids
            
    def init_buffer(self, init_optim_ids: Tensor):
        model = self.model
        tokenizer = self.tokenizer

        logger.info(f"Initializing attack buffer of size {self.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(self.buffer_size)

        init_buffer_ids = init_optim_ids

        true_buffer_size = max(1, self.buffer_size)

        # init_buffer_embeds = torch.cat([
        #     self.before_embeds.repeat(true_buffer_size, 1, 1),
        #     self.embedding_layer(init_buffer_ids),
        #     self.after_embeds.repeat(true_buffer_size, 1, 1),
        #     self.target_embeds.repeat(true_buffer_size, 1, 1),
        # ], dim=1)


        # TODO: call the self._compute_loss_batch method here
        init_buffer_losses = self._compute_loss_batch(init_buffer_ids, true_buffer_size)
        
        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer
    
    def _obtain_adversarial_query(self, query, optim_string):
       
        if self.optim_str_mode == "suffix":
            adv_query = query + optim_string
            
        elif self.optim_str_mode == "replace_query":
            adv_query = optim_string
            
        return adv_query
            
    
    @abstractmethod
    def _compute_gradients(self, optim_ids: Tensor):
        pass
    
    @abstractmethod
    def _compute_loss_batch(self, sampled_ids: Tensor, new_search_width: int):
        pass
    

    @staticmethod
    def _get_nonascii_toks(tokenizer, device="cpu"):
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
    
    def _get_bert_cosine_similarity(self, sentence1, sentence2):
        # Encode sentences with BERT tokenizer
        inputs1 = self.bert_tokenizer(sentence1, return_tensors="pt").to(self.bert_model.device)
        inputs2 = self.bert_tokenizer(sentence2, return_tensors="pt").to(self.bert_model.device)
        
        with torch.no_grad():
            outputs1 = self.bert_model.bert(**inputs1)
            outputs2 = self.bert_model.bert(**inputs2)
        
        # Get the [CLS] token embeddings as sentence representations
        cls_embedding1 = outputs1.last_hidden_state[:, 0, :]
        cls_embedding2 = outputs2.last_hidden_state[:, 0, :]
        
        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(cls_embedding1, cls_embedding2)
        
        return cosine_sim.item()
    
    def _get_sentence_transformer_cosine_similarity(self, sentence1, sentence2):
        # Encode sentences with SentenceTransformer
        embedding1 = self.sentence_transformer.encode(sentence1, convert_to_tensor=True)
        embedding2 = self.sentence_transformer.encode(sentence2, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        
        return cosine_sim.item()