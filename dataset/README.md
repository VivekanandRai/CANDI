# Datasets

Three grounded question-answering benchmarks, reformatted from their upstream releases into one common
record shape so the attack code can treat them interchangeably.

## Common schema

Every file is a **JSON array** of records. The attacks read exactly three fields:

| Field | Type | Meaning |
|---|---|---|
| `query` | string | The user question. This is what the attacks perturb. |
| `answer` | string | The reference answer, shown to the hallucination judge as ground truth. |
| `citations` | string | The retrieved context the model must stay faithful to. |

`idx` is carried through from upstream for traceability. Any other field is upstream metadata that this
codebase does not read.

Loading is `json.load` in both tracks — see `load_dataset` in
[`white_box_model_experiments/run.py`](../white_box_model_experiments/run.py) and
[`black_box_model_experiments/utils_openrouter.py`](../black_box_model_experiments/utils_openrouter.py).

## Files

### `failsafeqa_benchmark_data.json` — 100 records, 26 MB

Long-context financial QA over SEC filings. Derived from **FailSafeQA** (Writer Inc.).

This is the **first 100 of the 220 upstream records**, which is the slice the paper evaluates on. The
full set is ~54 MiB, over GitHub's 50 MiB warning threshold, and the contexts are very long (record 0's
context alone is ~128,000 characters of raw 10-K text).

Extra upstream fields retained but unused by this code: `tokens`, `context`, `citations_tokens`,
`ocr_context`, `error_query`, `incomplete_query`, `out-of-domain_query`, `out-of-scope_query`.

To rebuild from a full export — or to keep a different number of records:

```bash
python scripts/prepare_failsafeqa.py --source /path/to/full_failsafeqa.json --limit 100
python scripts/prepare_failsafeqa.py --source ... --limit 220           # the full set
python scripts/prepare_failsafeqa.py --source ... --slim                # drop ocr_context, ~13 MB
```

> Note: upstream ships this file with a `.jsonl` extension, but its contents are a standard JSON array,
> not one object per line. It is renamed to `.json` here to stop that tripping people up.

### `anah_benchmark_data.json` — 138 records, 293 KB

Wikipedia-grounded QA. A subset of **ANAH** (Analytical Annotation of Hallucinations, OpenDataLab /
InternLM). Fields: `idx`, `query`, `answer`, `citations`.

### `faitheval_openended_benchmark_data.json` — 1000 records, 2.4 MB

The **counterfactual** split of **FaithEval** (Salesforce AI Research) — contexts deliberately contradict
world knowledge, so a model that answers from parametric memory rather than the provided context is
detectably unfaithful. Fields: `idx`, `query`, `answer`, `citations`.

## Provenance and licensing

These files are **derivative reformattings of third-party benchmarks**, redistributed here so the
artifact is self-contained and runnable. They are not original contributions of this work.

Each upstream dataset carries its own licence, which governs your use of the corresponding file
regardless of this repository's MIT licence. Before redistributing or using these slices beyond
reproducing the paper, check the current terms on each upstream release:

| File | Upstream benchmark | Where to check terms |
|---|---|---|
| `failsafeqa_benchmark_data.json` | FailSafeQA (Writer Inc.) | The dataset's Hugging Face card |
| `anah_benchmark_data.json` | ANAH (OpenDataLab / InternLM) | The ANAH repository and dataset card |
| `faitheval_openended_benchmark_data.json` | FaithEval (Salesforce AI Research) | The FaithEval repository and dataset card |

If you use these datasets, cite the original benchmark papers in addition to this work.
