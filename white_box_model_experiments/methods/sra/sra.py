"""SRA (Search-based Red-Teaming Attack) scaffold for hallucination attacks.

Step 1: add class skeleton and configuration plumbing.
"""

from __future__ import annotations

from typing import Optional
import logging

from methods.base_attack_method import BaseAttackMethod
import torch
import torch.nn.functional as F
import random
import string


logger = logging.getLogger("sra")
if not logger.hasHandlers():
	_handler = logging.StreamHandler()
	_formatter = logging.Formatter(
		"%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	_handler.setFormatter(_formatter)
	logger.addHandler(_handler)
	logger.setLevel(logging.INFO)


class SRA(BaseAttackMethod):
	"""Search-based prompt expansion attack adapted for hallucination objectives."""

	def __init__(
		self,
		model,
		tokenizer,
		batch_size: int = 128,
		top_p: float = 0.99,
		top_k: int = 100,
		threshold: int = 1_000_000,
		response_length: int = 128,
		prompt_length_threshold: int = 1_000,
		stagnation_patience: int = 20,
		use_chat_template: bool = False,
		defense_type: Optional[str] = None,
		surrogate_model: Optional[object] = None,
		system_prompt: Optional[str] = None,
		prompt_format: Optional[str] = None,
	):
		super().__init__(model, tokenizer, system_prompt=system_prompt, prompt_format=prompt_format)

		self.batch_size = batch_size
		self.top_p = top_p
		self.top_k = top_k
		self.threshold = threshold
		self.response_length = response_length
		self.prompt_length_threshold = prompt_length_threshold
		self.stagnation_patience = stagnation_patience
		self.use_chat_template = use_chat_template
		self.defense_type = defense_type
		self.surrogate_model = surrogate_model
		self._validate_defense_type(defense_type)

		logger.info(
			"Initialized SRA: batch=%d top_p=%.3f top_k=%d threshold=%d",
			batch_size,
			top_p,
			top_k,
			threshold,
		)

	def run(self, input_context: str, query: str, expected_response: str):
		"""Run SRA search to find a hallucination-inducing prompt.

		Implementation is provided in later steps.
		"""
		self.stop_flag = False
		number_of_tried_prompts = 0
		final_result = None
		final_response = None
		final_hallucination = None
		final_justification = None
		prompt_length = 0
		stagnation_counter = 0
		stagnation_triggered = False
		last_pool_size = 0

		initial_prompt = self._build_full_prompt(input_context, query)
		# initial_response = self._generate_batch_responses([initial_prompt])[0]
		initial_hallucination_result = self.hallucination_check(
			input_context=input_context,
			query=query,
			expected_response=expected_response,
			# model_response=initial_response,
		)
		initial_response = initial_hallucination_result.get("model_response", "")
		if initial_hallucination_result.get("hallucination_detected") is True:
			return {
				"original_query": query,
				"adversarial_query": query,
				"steps": 0,
				"final_adversarial_response": initial_response,
				"final_hallucination": True,
				"final_hallucination_justification": initial_hallucination_result.get("justification", ""),
				"initial_response": initial_response,
				"initial_hallucination": True,
				"initial_hallucination_justification": initial_hallucination_result.get("justification", ""),
				"stagnation_triggered": False,
			}

		input_token_sequence = self._tokenize_prompt(query)
		last_pool_size = input_token_sequence.shape[0]

		for j in range(self.prompt_length_threshold):
			prompt_length = j
			updated_sequences = torch.tensor([], dtype=input_token_sequence.dtype).to(input_token_sequence.device)

			for k in range(0, len(input_token_sequence), self.batch_size):
				if number_of_tried_prompts >= self.threshold:
					self.stop_flag = True
					break

				batch = input_token_sequence[k : k + self.batch_size]
				prompt_text_batch = [self._decode_prompt_ids(seq) for seq in batch]
				number_of_tried_prompts += len(batch)

				full_prompts = [self._build_full_prompt(input_context, q) for q in prompt_text_batch]
				responses = self._generate_batch_responses(full_prompts)

				for adv_query, response in zip(prompt_text_batch, responses):
					hallucination_result = self.hallucination_check(
						input_context=input_context,
						query=adv_query,
						expected_response=expected_response,
						model_response=response,
					)
					if hallucination_result.get("hallucination_detected") is True:
						final_result = adv_query
						final_response = response
						final_hallucination = True
						final_justification = hallucination_result.get("justification")
						self.stop_flag = True
						break

				if self.stop_flag:
					break

				updated_sequences_k = self._update_sequences_and_probabilities(
					batch,
					self.top_p,
					self.top_k,
				)
				updated_sequences = torch.cat([updated_sequences, updated_sequences_k], dim=0)

			if self.stop_flag:
				break

			if updated_sequences.numel() == 0:
				break

			current_pool_size = updated_sequences.shape[0]
			if self.stagnation_patience > 0:
				if current_pool_size > last_pool_size:
					last_pool_size = current_pool_size
					stagnation_counter = 0
				else:
					stagnation_counter += 1

				if stagnation_counter >= self.stagnation_patience:
					logger.info(
						"Early stopping due to stagnation: pool size unchanged for %d iterations",
						self.stagnation_patience,
					)
					stagnation_triggered = True
					self.stop_flag = True
					input_token_sequence = updated_sequences.detach()
					break

			input_token_sequence = updated_sequences.detach()

		if final_result is None:
			final_result = self._decode_prompt_ids(input_token_sequence[-1])
			final_response = self.run_model_generation(input_context, final_result)
			final_hallucination_result = self.hallucination_check(
				input_context=input_context,
				query=final_result,
				expected_response=expected_response,
				model_response=final_response,
			)
			final_hallucination = final_hallucination_result.get("hallucination_detected", False)
			final_justification = final_hallucination_result.get("justification")

		return {
			"original_query": query,
			"adversarial_query": final_result,
			"steps": prompt_length + 1,
			"final_adversarial_response": final_response or "",
			"final_hallucination": bool(final_hallucination),
			"final_hallucination_justification": final_justification or "",
			"initial_response": initial_response,
			"initial_hallucination": initial_hallucination_result.get("hallucination_detected", False),
			"initial_hallucination_justification": initial_hallucination_result.get("justification", ""),
			"stagnation_triggered": stagnation_triggered,
		}

	def _tokenize_prompt(self, prompt_text: str) -> torch.Tensor:
		"""Tokenize a single prompt into input_ids on the model device."""
		inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
		return inputs["input_ids"].to(self.model.device)

	def _decode_prompt_ids(self, input_ids: torch.Tensor) -> str:
		"""Decode a single prompt sequence to text."""
		return self.tokenizer.decode(input_ids, skip_special_tokens=True)

	def _update_sequences_and_probabilities(
		self,
		input_token_sequence_batch: torch.Tensor,
		top_p: float,
		top_k: int,
	) -> torch.Tensor:
		"""Expand a batch of prompt ids by one token using top-p/top-k filtering."""
		attention_mask_batch = torch.ones_like(input_token_sequence_batch)
		with torch.no_grad():
			output = self.generation_model.generate(
				input_token_sequence_batch,
				attention_mask=attention_mask_batch,
				return_dict_in_generate=True,
				max_length=1 + input_token_sequence_batch.shape[-1],
				min_length=1 + input_token_sequence_batch.shape[-1],
				output_scores=True,
			)
		logits = torch.stack(output.scores)
		logits = torch.permute(logits, (1, 0, 2))
		probabilities = F.softmax(logits, dim=-1)
		return self._update_token_sequence_and_probabilities(
			probabilities,
			top_p,
			top_k,
			input_token_sequence_batch,
		)

	def _update_token_sequence_and_probabilities(
		self,
		probabilities: torch.Tensor,
		top_p: float,
		top_k: int,
		input_token_sequence_batch: torch.Tensor,
	) -> torch.Tensor:
		"""Apply nucleus sampling mask and return expanded prompt sequences."""
		probabilities_sort, probabilities_idx = torch.sort(probabilities, dim=-1, descending=True)
		probabilities_sum = torch.cumsum(probabilities_sort, dim=-1)
		mask = probabilities_sum - probabilities_sort < top_p
		top_p_tokens = probabilities_idx[mask]
		token_counter = 0
		updated_sequences = torch.tensor([], dtype=input_token_sequence_batch.dtype).to(probabilities.device)

		for i in range(probabilities.size(0)):
			candidate_tokens_number = mask[i].sum().item()
			if candidate_tokens_number <= top_k:
				expanded_sequences = input_token_sequence_batch[i : i + 1].expand(candidate_tokens_number, -1)
				top_p_tokens_ = top_p_tokens[token_counter : token_counter + candidate_tokens_number].unsqueeze(1)
			else:
				expanded_sequences = input_token_sequence_batch[i : i + 1].expand(top_k, -1)
				top_p_tokens_ = top_p_tokens[token_counter : token_counter + top_k].unsqueeze(1)

			updated_sequences_i = torch.cat([expanded_sequences, top_p_tokens_], dim=-1)
			updated_sequences = torch.cat([updated_sequences, updated_sequences_i], dim=0)
			token_counter += candidate_tokens_number

		return updated_sequences

	def _build_full_prompt(self, input_context: str, query: str) -> str:
		"""Build the full prompt with or without chat template."""
		query = self._apply_defense_perturbation(query)
		if self.use_chat_template:
			return self.apply_chat_template_to_context_query(input_context, query)
		return self.prompt_format.format(context=input_context, query=query)

	def _generate_batch_responses(self, prompt_text_batch: list[str]) -> list[str]:
		"""Generate responses for a batch of prompts."""
		inputs = self.tokenizer(prompt_text_batch, return_tensors="pt", padding=True, truncation=True)
		input_ids = inputs["input_ids"].to(self.model.device)
		attention_mask = inputs["attention_mask"].to(self.model.device)
		with torch.no_grad():
			generated_ids = self.generation_model.generate(
				input_ids,
				attention_mask=attention_mask,
				max_new_tokens=self.response_length,
				do_sample=False,
				pad_token_id=self.tokenizer.pad_token_id,
			)

		responses: list[str] = []
		for i in range(generated_ids.shape[0]):
			prompt_len = int(attention_mask[i].sum().item())
			response_ids = generated_ids[i][prompt_len:]
			responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
		return responses

	def _validate_defense_type(self, defense_type: Optional[str]) -> None:
		valid = {None, "random_swap", "random_patch", "random_insert"}
		if defense_type not in valid:
			raise ValueError(f"Unsupported defense_type: {defense_type}. Valid options: {sorted(v for v in valid if v is not None)} or None")

	def _apply_defense_perturbation(self, prompt_text: str) -> str:
		if self.defense_type is None:
			return prompt_text
		if self.defense_type == "random_swap":
			return self._random_swap(prompt_text)
		if self.defense_type == "random_patch":
			return self._random_patch(prompt_text)
		if self.defense_type == "random_insert":
			return self._random_insert(prompt_text)
		return prompt_text

	def _random_swap(self, text: str, q: int = 10) -> str:
		"""Randomly replace ~q% of characters."""
		if not text:
			return text
		list_s = list(text)
		num = max(1, int(len(text) * q / 100))
		sampled_indices = random.sample(range(len(text)), min(num, len(text)))
		for i in sampled_indices:
			list_s[i] = random.choice(string.printable)
		return "".join(list_s)

	def _random_patch(self, text: str, q: int = 10) -> str:
		"""Randomly replace a contiguous substring (~q% length)."""
		if not text:
			return text
		substring_width = max(1, int(len(text) * q / 100))
		max_start = max(0, len(text) - substring_width)
		start_index = random.randint(0, max_start) if max_start > 0 else 0
		sampled_chars = "".join(random.choice(string.printable) for _ in range(substring_width))
		list_s = list(text)
		list_s[start_index:start_index + substring_width] = sampled_chars
		return "".join(list_s)

	def _random_insert(self, text: str, q: int = 10) -> str:
		"""Randomly insert ~q% extra characters."""
		if not text:
			return text
		list_s = list(text)
		num = max(1, int(len(text) * q / 100))
		sampled_indices = random.sample(range(len(text)), min(num, len(text)))
		for i in sampled_indices:
			list_s.insert(i, random.choice(string.printable))
		return "".join(list_s)
