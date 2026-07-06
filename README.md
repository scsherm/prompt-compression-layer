# Lingua Symbolic Prompt Compiler

Tokenizer-aware lingua-symbolic prompt compiler.

The compiler takes a target model, an original prompt template, and input examples. It records behavioral reference outputs from the original prompt, then searches for shorter prompt templates whose target-model outputs stay close to those references.

```text
target model M
original prompt template P
input examples X

reference: y_i = M(P, x_i)
candidate: yhat_i = M(P_candidate, x_i)

objective: reduce instruction tokens while preserving target-model behavior
```

## Current Pipeline

1. Build behavioral references by running the frozen target model over the original prompt and input set.
2. Split references into train, dev, and holdout sets.
3. Extract bound prompt variables such as `{{input}}`, `{{query}}`, or `{{document}}` as protected slot chunks.
4. Generate chunking variants around those bound slots with paragraph, sentence, markdown, schema-aware, instruction-role, and token-window chunkers.
5. Use rewrite operators to produce candidate chunk variants.
6. Assemble candidate prompt templates while preserving the original placeholder sequence.
7. Run each candidate through the target model on the current evaluation subset.
8. Score candidates by instruction-token reduction and Euclidean embedding drift, with optional equivalence scorers such as an LLM judge, ROUGE, or BLEU.
9. Keep a Pareto frontier, generate the next population from frontier candidates, then evaluate finalists on dev and holdout sets.
10. Write prompts, traces, reports, frontier files, candidate outputs, and run logs to disk.

## Model Roles

There are two model roles and three LLM call sites.

The target behavior model is the user-selected `--model`. It is used for both behavior-bearing call sites:

- reference generation: run the original prompt over each input
- candidate completion: run each compressed candidate prompt over the same inputs

The proposer model is the rewrite model. It rewrites individual non-bound chunks into shorter variants. It does not define the behavioral reference. For OpenAI-backed runs, the default proposer is:

```text
gpt-5.4-mini-2026-03-17
```

The target behavior model remains a CLI input. The proposer model is configurable with `--proposer-model`.

## Bound Prompt Variables

Prompt variables use double-curly template slots:

```text
{{input}}
{{query}}
{{document}}
{{tone}}
```

These slots form a bound variable layer. Chunkers split around them, mark them as protected input-slot chunks, and candidate assembly preserves their names and order. The proposer receives only rewriteable chunks; protected chunks pass through unchanged.

The current assembly path checks the placeholder sequence before returning a candidate:

```text
original placeholders == assembled placeholders
```

## Rewrite Operators

The current exploration space includes:

- keep
- short English
- telegraph English
- symbolic DSL
- schema abbreviation
- hybrid symbolic English
- short Mandarin
- formal Chinese
- classical-Chinese-like
- Mandarin-symbolic
- bilingual DSL
- mixed minimum-token form

The tokenizer decides whether Mandarin, symbolic, English, or mixed forms actually reduce cost for the configured model/tokenizer.

## Setup

Create the project Conda environment:

```bash
conda env create -f environment.yml
conda activate prompt-compression-layer
```

The environment used on this machine is:

```text
/Users/scsherm/anaconda3/envs/prompt-compression-layer
```

If `conda run` resolves the wrong interpreter, use the direct Python path:

```bash
/Users/scsherm/anaconda3/envs/prompt-compression-layer/bin/python
```

Pip-only setup:

```bash
python3 -m pip install -r requirements.txt
```

`OPENAI_API_KEY` is loaded from the environment or `.env.local`.

## CLI

Mock local run:

```bash
python3 -m prompt_compiler.cli \
  --provider mock \
  --model mock \
  --prompt examples/original_prompt.txt \
  --inputs examples/inputs.jsonl \
  --output-dir runs/mock_run \
  --epochs 3 \
  --population-size 32 \
  --tokenizer approx \
  --embedding-provider lexical \
  --require-json
```

OpenAI target model with OpenAI proposer and Mixedbread embeddings:

```bash
/Users/scsherm/anaconda3/envs/prompt-compression-layer/bin/python -m prompt_compiler.cli \
  --provider openai \
  --model gpt-5-nano-2025-08-07 \
  --proposer openai \
  --proposer-model gpt-5.4-mini-2026-03-17 \
  --proposer-reasoning-effort medium \
  --proposer-max-output-tokens 16384 \
  --prompt examples/no_robots_rich_prompt.txt \
  --inputs data/hf/no_robots_100.jsonl \
  --output-dir runs/no_robots_rich_gpt5nano_llm \
  --epochs 4 \
  --population-size 16 \
  --max-output-tokens 256 \
  --max-concurrency 1 \
  --tokenizer tiktoken:cl100k_base \
  --embedding-provider sentence-transformers \
  --embedding-model mixedbread-ai/mxbai-embed-large-v1 \
  --require-json
```

Useful CLI controls:

- `--chunkers`: comma-separated chunkers, default `paragraph,sentence,markdown,schema_aware`
- `--live-log-file`: stable JSONL mirror for `tail -f`, default `runs/live_run_events.jsonl`
- `--max-concurrency`: maximum concurrent candidate evaluations
- `--input-limit`: use the first N rows from the input JSONL
- `--send-openai-sampling-params`: sends `temperature` and `top_p` to OpenAI Responses for endpoints that support them

## Tokenizers

- `approx`: dependency-free tokenizer for local development
- `tiktoken:<encoding>`: explicit tiktoken encoding, for example `tiktoken:cl100k_base`
- `model:<model-name>`: `tiktoken.encoding_for_model(...)`

The primary compression metric is instruction-token reduction:

```text
1 - candidate_instruction_tokens / original_instruction_tokens
```

Rendered prompt token counts are also tracked because long inputs can dominate total tokens.

## Embedding Drift

Embedding drift is Euclidean distance over completion embeddings:

```text
drift = || embed(candidate_output) - embed(reference_output) ||_2
```

Supported providers:

- `lexical`: dependency-free local fallback
- `sentence-transformers`: local sentence-transformers embeddings
- `hf-inference`: Hugging Face Inference feature extraction

The default embedding model for non-lexical providers is:

```text
mixedbread-ai/mxbai-embed-large-v1
```

## Evaluation Objective

The core evaluation objective is the tradeoff between compression and generated-output equivalence:

```text
maximize token_reduction
minimize embedding_drift
```

When a scalar objective is useful, use token ratio plus Euclidean embedding drift:

```text
objective = lambda_token * token_ratio + lambda_embed * embedding_drift
```

Optional equivalence signals can be added as additional output-distance terms:

- separate LLM judge disagreement
- ROUGE distance
- BLEU distance

Structured output validity is tracked separately from the scalar objective:

- format failure rate
- task-field failure rate
- failure cases

Those validity signals describe whether a candidate output remains usable by the downstream contract. They are not output-equivalence distance terms.

## Hugging Face Data

The repo includes a normalized sample from `HuggingFaceH4/no_robots`:

```text
data/hf/no_robots_100.jsonl
```

Refresh it with:

```bash
/Users/scsherm/anaconda3/envs/prompt-compression-layer/bin/python \
  scripts/download_instruction_dataset.py \
  --dataset HuggingFaceH4/no_robots \
  --config default \
  --split train \
  --limit 100 \
  --output data/hf/no_robots_100.jsonl
```

Each row contains `id`, `input`, `reference_output`, `dataset`, `category`, and source metadata.

## Output Artifacts

Each run writes artifacts under `--output-dir`:

- `best_prompt.txt`
- `best_prompt_template.json`
- `compression_report.json`
- `pareto_frontier.csv`
- `dev_frontier.csv`
- `failures.json`
- `reference_dataset.jsonl`
- `candidate_prompts.jsonl`
- `candidate_reports.jsonl`
- `candidate_outputs.jsonl`
- `holdout_reports.jsonl`
- `run_events.jsonl`
- `proposer_traces.jsonl` when using the LLM proposer

The stable live log mirror defaults to:

```text
runs/live_run_events.jsonl
```
