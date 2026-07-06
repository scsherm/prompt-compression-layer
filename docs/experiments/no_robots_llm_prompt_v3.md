# no_robots_llm_prompt_v3 Checkpoint

Checkpoint date: 2026-07-06

## Purpose

This checkpoint validates the current end-to-end compression path on a small no-robots sample:

```text
original prompt + inputs
-> LLM proposer chunk rewrites
-> candidate prompt templates
-> rendered candidate prompts
-> GPT-5 Nano completions
-> reference comparisons
-> normalized loss metrics
```

The run is a small-sample verification baseline, not a general benchmark.

## Configuration

Candidate generation:

```text
run directory: runs/no_robots_llm_prompt_v3
prompt: examples/no_robots_rich_prompt.txt
inputs: data/hf/no_robots_100.jsonl
proposer model: gpt-5.4-mini-2026-03-17
target model context: gpt-5-nano-2025-08-07
generated candidates: 8
previewed candidates: 6
minimum preview token reduction: 0.15
chunkers: paragraph,instruction_role
```

Candidate completion comparison:

```text
run directory: runs/no_robots_llm_prompt_v3_side_by_side_v3
target model: gpt-5-nano-2025-08-07
inputs evaluated: 3
reference completions: 3
candidate completions: 18
total target-model calls: 21
max output tokens: 2048
reasoning effort: minimal
```

Embedding and loss:

```text
embedding model: mixedbread-ai/mxbai-embed-large-v1
embedding distance: raw Euclidean distance
semantic_drift_normalization: 10.578125
semantic_drift_norm: clamp(semantic_drift / semantic_drift_normalization, 0, 1)
token_reduction_norm: clamp(token_reduction, 0, 1)
loss: 0.5 * semantic_drift_norm + 0.5 * (1 - token_reduction_norm)
direction: lower is better
range: 0..1
```

The semantic-drift normalization value is the maximum observed raw pairwise drift in this checkpoint's candidate-completion comparison set.

## Artifacts

Candidate generation artifacts:

```text
runs/no_robots_llm_prompt_v3/original_prompt.txt
runs/no_robots_llm_prompt_v3/chunking_plan.json
runs/no_robots_llm_prompt_v3/proposer_traces.jsonl
runs/no_robots_llm_prompt_v3/candidate_templates.jsonl
runs/no_robots_llm_prompt_v3/candidate_templates.md
```

Completion comparison and audit artifacts:

```text
runs/no_robots_llm_prompt_v3_side_by_side_v3/side_by_side.md
runs/no_robots_llm_prompt_v3_side_by_side_v3/side_by_side.jsonl
runs/no_robots_llm_prompt_v3_side_by_side_v3/summary.json
runs/no_robots_llm_prompt_v3_side_by_side_v3/provenance_audit.md
runs/no_robots_llm_prompt_v3_side_by_side_v3/provenance_audit.json
runs/no_robots_llm_prompt_v3_side_by_side_v3/openai_proposer_retrieval_audit.json
runs/no_robots_llm_prompt_v3_side_by_side_v3/openai_response_retrieval_audit.json
runs/no_robots_llm_prompt_v3_side_by_side_v3/loss_metrics.md
runs/no_robots_llm_prompt_v3_side_by_side_v3/loss_metrics.json
```

The `runs/` directory is ignored by git, so these files are local run artifacts.

## Verification

Proposer-stage verification:

- `24` saved proposer response IDs were retrieved from OpenAI.
- All retrieved proposer responses had status `completed`.
- All retrieved proposer response text matched the saved `proposer_traces.jsonl` text by SHA-256.

Completion-stage verification:

- `21` saved target-model response IDs were retrieved from OpenAI.
- All retrieved target-model responses had status `completed`.
- All retrieved target-model output text matched the saved side-by-side artifact text by SHA-256.

Prompt-path verification:

- `18` candidate completion rows were present.
- `6` candidate templates were present.
- `3` inputs were present.
- Candidate IDs in the side-by-side artifact matched the candidate generation artifact.
- Candidate prompt templates in the side-by-side artifact matched the source candidate template file.
- Each rendered candidate prompt differed from the rendered original prompt.
- Each rendered candidate prompt contained the exact input once.
- Candidate input-token usage was lower than reference input-token usage for every evaluated pair.

## Candidate Ranking

| rank | candidate | candidate id | token reduction | avg drift | avg normalized drift | loss | validation note |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 4 | `023639879784` | 0.399 | 6.3216 | 0.5976 | 0.5994 | clear |
| 2 | 3 | `6af23f6fc2b3` | 0.416 | 6.5130 | 0.6157 | 0.5998 | clear |
| 3 | 1 | `4510ee94bcee` | 0.347 | 6.1029 | 0.5769 | 0.6151 | clear |
| 4 | 5 | `4ec74857c672` | 0.387 | 7.0729 | 0.6686 | 0.6407 | clear |
| 5 | 2 | `1d03bbdd31de` | 0.382 | 7.2552 | 0.6859 | 0.6522 | clear |
| 6 | 6 | `8072d6c59e6d` | 0.295 | 7.7018 | 0.7281 | 0.7166 | meta-structure leakage |

Candidate 4 is the current best candidate under the normalized loss on this 3-input checkpoint, with Candidate 3 effectively tied.

## Interpretation

The checkpoint shows the current pipeline can generate compressed candidate prompt templates, render them with preserved placeholders, produce target-model completions, and rank candidates with a normalized loss that combines token reduction and semantic drift.

Candidate 6 demonstrates why validation checks should remain separate from loss: it leaks meta-structure into generated answers on two evaluated inputs. That behavior should be flagged or filtered independently rather than folded into semantic drift.

## Next Steps

- Evaluate the best candidates on a larger input subset.
- Promote the side-by-side completion comparison into a first-class CLI/reporting path.
- Add a candidate-output validation check for proposer/meta-structure leakage.
- Verify normalized drift and normalized loss reporting on the next larger optimization run.
