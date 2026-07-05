# Prompt Compression Layer

Tokenizer-aware behavioral prompt compression compiler.

The compiler takes a target model client, an original prompt template, and input examples. It first records behavioral reference outputs from the original prompt, then searches for shorter prompt templates whose outputs stay close to those references.

The search space intentionally includes compact English, symbolic DSL, schema abbreviations, short Mandarin, classical-Chinese-like directives, Mandarin-symbolic hybrids, bilingual DSL, and mixed dense forms. The tokenizer and evaluator decide what survives.

## Minimal CLI Run

```bash
python3 -m prompt_compiler.cli \
  --model mock \
  --prompt prompts/original.txt \
  --inputs data/inputs.jsonl \
  --output-dir runs/run_001 \
  --epochs 3 \
  --population-size 32 \
  --tokenizer approx \
  --embedding-provider lexical \
  --require-json
```

The built-in CLI uses a deterministic mock model. Real model use is via the `ModelClient` protocol so callers can freeze model name, version, tokenizer, generation params, tools, and system prompt.

Optional adapters:

```bash
conda env create -f environment.yml
conda activate prompt-compression-layer
export HF_TOKEN=...
```

This environment is project-specific and is expected at:

```text
/Users/scsherm/anaconda3/envs/prompt-compression-layer
```

If `conda run -n prompt-compression-layer ...` resolves the wrong Python on this machine, use the direct interpreter:

```bash
/Users/scsherm/anaconda3/envs/prompt-compression-layer/bin/python -m unittest discover -v
```

For pip-only setup:

```bash
python3 -m pip install -r requirements.txt
```

Tokenizer specs:

- `approx`: dependency-free local tokenizer for development.
- `tiktoken:<encoding>`: optional `tiktoken` adapter, for example `tiktoken:cl100k_base`.
- `model:<model-name>`: optional `tiktoken.encoding_for_model(...)` adapter.

Embedding drift providers:

- `lexical`: dependency-free local fallback.
- `sentence-transformers`: local embeddings with `sentence-transformers`.
- `hf-inference`: Hugging Face `InferenceClient.feature_extraction(...)`.

The default embedding model for non-lexical providers is `mixedbread-ai/mxbai-embed-large-v1`, a Hugging Face sentence-transformers-compatible feature-extraction model: <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>.

Embedding drift is Euclidean distance over embedding vectors:

```text
drift = || embed(candidate_output) - embed(reference_output) ||2
```

Example with Hugging Face provider-backed Mixedbread embeddings:

```bash
python3 -m prompt_compiler.cli \
  --model mock \
  --prompt examples/original_prompt.txt \
  --inputs examples/inputs.jsonl \
  --output-dir runs/hf_embeddings \
  --embedding-provider hf-inference \
  --embedding-model mixedbread-ai/mxbai-embed-large-v1
```

## Core Artifacts

- `best_prompt.txt`
- `best_prompt_template.json`
- `compression_report.json`
- `pareto_frontier.csv`
- `failures.json`
- `reference_dataset.jsonl`
- `candidate_reports.jsonl`
- `candidate_outputs.jsonl`
