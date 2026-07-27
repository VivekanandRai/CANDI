"""Vanilla GCG implementation using the new BaseGCG interface.

Step 2: add prompt-splitting/combining and initialization helpers.
"""

import logging
import torch
import gc

from methods.gcg.base_gcg import BaseGCG

logger = logging.getLogger("vanilla_gcg")
if not logger.hasHandlers():
	handler = logging.StreamHandler()
	formatter = logging.Formatter(
		"%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)


class VanillaGCG(BaseGCG):
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
		n_replace: int = 1,
		allow_non_ascii: bool = False,
		verbose: bool = False,
		stagnation_patience: int = 20,
		perplexity_threshold: float | None = None,
	):
		super().__init__(
			model,
			tokenizer,
			top_k=top_k,
			num_steps=num_steps,
			search_width=search_width,
			optim_str_mode=optim_str_mode,
			optim_str_constraint=optim_str_constraint,
			system_prompt=system_prompt,
			prompt_format=prompt_format,
			optim_str_length=optim_str_length,
			batch_size=batch_size,
			buffer_size=buffer_size,
			allow_non_ascii=allow_non_ascii,
			n_replace=n_replace,
			verbose=verbose,
			stagnation_patience=stagnation_patience,
			perplexity_threshold=perplexity_threshold,
		)

		

		
	# def _split_context_prompt_into_parts(self, context: str, query: str):
	# 	"""Build chat prompt and split on `{optim_str}` placeholder."""
	# 	full_prompt = self.apply_chat_template_to_context_query(context, query)
	# 	placeholder = "{optim_str}"
	# 	if placeholder not in full_prompt:
	# 		full_prompt = full_prompt + placeholder
	# 	try:
	# 		before_str, after_str = full_prompt.split(placeholder)
	# 	except ValueError as exc:  # pragma: no cover - defensive guard
	# 		raise RuntimeError("Unable to split prompt on {optim_str} placeholder") from exc
	# 	return before_str, after_str

	def _combine_parts_into_context_prompt(self, before_str: str, optim_str: str, after_str: str):
		return before_str + optim_str + after_str


	def run(self, input_context: str, query: str, expected_response: str, target: str):
		"""Override to accept `target`; implementation will follow in later steps."""
		
		target_ids = self.tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.generation_model.device, torch.int64)
		target_embeds = self.embedding_layer(target_ids)
		self.target_ids = target_ids
		self.target_embeds = target_embeds
  
		return super().run(input_context, query, expected_response)


	def _compute_gradients(self, optim_ids: torch.Tensor):
		model = self.generation_model
		embedding_layer = self.embedding_layer

		# Create the one-hot encoding matrix of our optimized token ids
		optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
		optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
		optim_ids_onehot.requires_grad_()

		# (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
		optim_embeds = optim_ids_onehot @ embedding_layer.weight

		if self.prefix_cache:
			input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
			output = model(
				inputs_embeds=input_embeds,
				past_key_values=self.prefix_cache,
				use_cache=True,
			)
		else:
			input_embeds = torch.cat(
				[
					self.before_embeds,
					optim_embeds,
					self.after_embeds,
					self.target_embeds,
				],
				dim=1,
			)
			output = model(inputs_embeds=input_embeds)

		logits = output.logits

		# Shift logits so token n-1 predicts token n
		shift = input_embeds.shape[1] - self.target_ids.shape[1]
		shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
		shift_labels = self.target_ids
		
		loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

		optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

		return optim_ids_onehot_grad

	def _compute_loss_batch(self, sampled_ids, new_search_width):
		all_loss = []
		prefix_cache_batch = []

		if self.prefix_cache:
			input_embeds = None
			raise NotImplementedError("Prefix cache batching not implemented yet.")

		else:
  
			input_embeds = torch.cat([
				self.before_embeds.repeat(new_search_width, 1, 1),
				self.embedding_layer(sampled_ids),
				self.after_embeds.repeat(new_search_width, 1, 1),
				self.target_embeds.repeat(new_search_width, 1, 1),
			], dim=1)

		for i in range(0, input_embeds.shape[0], self.batch_size):
			with torch.no_grad():
				input_embeds_batch = input_embeds[i:i + self.batch_size]
				current_batch_size = input_embeds_batch.shape[0]

				if self.prefix_cache:
					if not prefix_cache_batch or current_batch_size != self.batch_size:
						prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

					outputs = self.generation_model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True)
				else:
					outputs = self.generation_model(inputs_embeds=input_embeds_batch)

				logits = outputs.logits

				tmp = input_embeds.shape[1] - self.target_ids.shape[1]
				shift_logits = logits[..., tmp-1:-1, :].contiguous()
				shift_labels = self.target_ids.repeat(current_batch_size, 1)

				
				loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

				loss = loss.view(current_batch_size, -1).mean(dim=-1)
				all_loss.append(loss)

				# if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
				# 	print("Early stopping: target matched exactly.")
				# 	self.stop_flag = True

				del outputs
				gc.collect()
				torch.cuda.empty_cache()

		return torch.cat(all_loss, dim=0)