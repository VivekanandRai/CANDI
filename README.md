# Eliciting Intrinsic Hallucinations in Large Language Models

Code and data for the COLM 2026 paper *Eliciting Intrinsic Hallucinations in Large Language Models*.

Large language models are frequently paired with external knowledge sources such as RAG to improve factual accuracy and reduce hallucination.
Such systems nevertheless remain susceptible to *intrinsic* hallucinations, in which the model generates
unfaithful or fabricated information that is not supported by the retrieved evidence.

Our work proposes a framework for assessing model robustness against this failure mode by
stress-testing generators with natural, semantically equivalent variations of a user query, discovered
via adversarial optimisation. The framework couples an intrinsic-hallucination objective with strict
semantic-equivalence constraints, and is instantiated across a range of adversarial attack techniques in
white-box, gray-box, and black-box threat models. Evaluating these attacks on 5 open-source and 5
closed-source generator models across 3 datasets shows that even state-of-the-art models are highly
susceptible to meaning-preserving perturbations, which degrade contextual faithfulness by up to 70.3%
for `gpt-5-mini`. These results indicate that faithful use of in-context evidence remains fragile even
in state-of-the-art LLMs, motivating architectures and training objectives that enforce robust grounding
independent of surface query form.

---

## Contents

| Track | Threat model | Attacks | Entrypoint |
|---|---|---|---|
| **White-box** | Full access to weights, gradients and logits of a local Hugging Face model | GCG, AutoDAN, SRA, PAIR, SECA | [`white_box_model_experiments/run.py`](white_box_model_experiments/run.py) |
| **Black-box** | Query-only access to an API model via OpenRouter | PAIR, SECA | [`run_pair.py`](black_box_model_experiments/run_pair.py), [`run_seca.py`](black_box_model_experiments/run_seca.py) |

---

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/atriviveksharma/intrinsic_hall.git
cd intrinsic_hall

# 1. Create the environment from the committed lockfile
uv sync

# 2. Add your API keys
cp .env.example .env    # then fill in HF_TOKEN and OPENROUTER_API_KEY
```

## Running Experiments



```bash
# Black-box: PAIR against gpt-5-nano on ANAH (needs OPENROUTER_API_KEY)
uv run python black_box_model_experiments/run_pair.py \
    --config black_box_model_experiments/experiment_configs/pair_openai_gpt-5-nano_anah.json

# White-box: PAIR against a local Llama-3.2-1B (needs a GPU and HF_TOKEN)
uv run python white_box_model_experiments/run.py \
    --config white_box_model_experiments/configs/llama3.2-1b-vanilla-pair.json
```

Every runner accepts `--dry-run`, which validates the config, resolves the dataset and output paths,
prints the resolved plan and exits without calling a model. Use it to check a config before committing
GPU hours or API spend.

### API keys

| Variable | Needed by |
|---|---|
| `OPENROUTER_API_KEY` | All black-box experiments, **and all white-box runs** — the hallucination judge, semantic-equivalence judge and embedding calls all go through OpenRouter. |
| `HF_TOKEN` | White-box only, to download gated Hugging Face checkpoints. |

Both are read from `.env` automatically, or from the environment if already exported.


## Reproducing the paper

**White-box** — 5 attacks × 5 open-weight models × 3 datasets. The repository ships one example config
per attack (`llama3.2-1b` × `failsafeqa`); generate the rest with:

```bash
uv run python scripts/generate_configs.py white-box \
    --models llama3.2-1b llama3.2-3b llama3.1-8b qwen3-4b gemma3-1b \
    --datasets failsafeqa anah faitheval \
    --attacks vanilla_gcg_synonym vanilla_autodan_synonym sra_suffix_3 vanilla_pair vanilla_seca
```

**Black-box** — 2 attacks × 5 API models × 3 datasets. All 30 configs are checked in under
`black_box_model_experiments/experiment_configs/`, covering `openai/gpt-5-nano`, `openai/gpt-5-mini`,
`openai/gpt-5-chat`, `google/gemini-2.5-flash-lite` and `minimax/minimax-m2.1`.

## Citation

```bibtex
@inproceedings{sharma2026intrinsic,
  title     = {Eliciting Intrinsic Hallucinations in Large Language Models},
  author    = {Sharma, Atri},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## Licence

Code released under the [MIT Licence](LICENSE). The benchmark slices in `dataset/` are derived from
third-party datasets under their own terms — see [dataset/README.md](dataset/README.md).