# Lingua Symbolic Prompt Compiler

This project learns shorter reusable instruction prompts from the behavior of a target model.

Given an original prompt `P`, target model `M`, and completion inputs `x_i`, it first records behavioral references:

```text
y_i = M(P, x_i)
```

It then searches for complete compressed prompts `P'` that save instruction tokens while keeping the resulting completions close to those references:

```text
maximize instruction-token savings
minimize behavior loss between M(P', x_i) and y_i
```

The original prompt defines the reference behavior for the search.

## Optimization loop

The active optimizer is a feedback-conditioned, full-prompt black-box search:

1. Run the original prompt on every input to create the reference completions.
2. Ask the proposer model for a diverse batch of complete, shorter prompt templates.
3. Measure every proposal with the target tokenizer and run it on the same completion inputs.
4. Compare candidate completions with the references using semantic distance and any configured output-contract checks.
5. Maintain a cross-round Pareto archive over token savings and behavior loss.
6. Feed frontier prompts, actual token counts, prompt diffs, poor trials, and worst completion residuals into the next proposer call.
7. Repeat only candidates whose observed rewards are close or uncertain.
8. Stop when the Pareto frontier no longer materially improves, or when the round budget is reached.
9. Re-evaluate frontier candidates on dev and holdout examples.

The proposer chooses the rewrite scope itself. A proposal can reorganize the whole prompt, merge distant redundancy, replace a section, or make a small repair.

## Behavior reward

Candidate quality is learned from observed target-model completions.

When a dataset row contains a labeled JSON `expected` value, the optimizer measures field/value true positives, false positives, false negatives, precision, recall, F1, exact match, JSON validity, and schema validity. Candidate reward measures the change in labeled task quality relative to the original prompt. The same field/value comparison supports extraction tasks such as malicious-IP identification and other structured-output contracts.

When no task label is available, the soft behavior loss is:

```text
residual semantic distance
+ format failure rate
+ task-field failure rate
```

Natural target-model variation is estimated by repeating the original prompt. For labeled tasks this measures task-quality variation; otherwise it measures semantic-output variation. The observed variance is used to decide which close candidates need another rollout.

Format and task checks contribute continuously to behavior loss. Candidate templates are eligible when:

- it must use fewer instruction tokens than the original;
- it must preserve the template placeholder sequence, such as `{{input}}`.

## Model roles

- The target model produces both the original reference completions and candidate completions.
- The proposer model uses measured search history to propose the next full-prompt batch.
- The configured tokenizer supplies instruction-token counts.

## Setup

```bash
conda env create -f environment.yml
conda activate prompt-compression-layer
```

or:

```bash
python3 -m pip install -r requirements.txt
```

The CLI loads `OPENAI_API_KEY` from the environment or `.env.local`.

## Run

```bash
python3 -m prompt_compiler.cli \
  --provider openai \
  --model gpt-5-nano \
  --proposer-model gpt-5.4-mini \
  --prompt examples/no_robots_rich_prompt.txt \
  --inputs data/hf/no_robots_100.jsonl \
  --output-dir runs/no_robots_feedback \
  --rounds 8 \
  --batch-size 8 \
  --convergence-patience 3 \
  --max-candidate-rollouts 2 \
  --tokenizer model:gpt-5-nano \
  --embedding-provider sentence-transformers \
  --embedding-model mixedbread-ai/mxbai-embed-large-v1 \
  --require-json
```

The target may also be the deterministic local mock while the proposal policy remains an LLM:

```bash
python3 -m prompt_compiler.cli \
  --provider mock \
  --model mock \
  --prompt examples/original_prompt.txt \
  --inputs examples/inputs.jsonl \
  --output-dir runs/mock_feedback \
  --rounds 3 \
  --batch-size 4
```

Useful controls:

- `--frontier-parent-limit`: number of diverse Pareto parents shown to the proposer.
- `--recent-contrast-limit`: number of poor recent trials shown as counterexamples.
- `--worst-example-limit`: worst completion residuals included per feedback candidate.
- `--baseline-repeats`: original-prompt repeats used to estimate output noise.
- `--repeat-top-k`: maximum close or uncertain candidates selected for extra rollouts.
- `--max-candidate-rollouts`: total rollout cap for those selected candidates.
- `--min-frontier-improvement` and `--convergence-patience`: convergence controls.
- `--preview-proposals`: generate the first full-prompt proposal batch without target evaluation.
- `--no-feedback`: withhold candidate outcomes from later rounds for an ablation.
- `--selection-behavior-penalty`: explicit behavior-loss penalty used only to recommend one point from the final Pareto frontier.

The default `auto` evaluation profile uses labeled precision/recall/F1 when `expected` JSON is present and normalized sentence-transformer embeddings for unlabeled tasks. Lexical distance is available for offline evaluation.

The default `auto` tokenizer uses the target model's tokenizer for OpenAI runs and the approximate tokenizer for local mock runs.

## Budget-matched feedback ablation

Run the same round, batch, rollout, model, and dataset settings twice, changing only `--no-feedback`:

```bash
# learned feedback loop
python3 -m prompt_compiler.cli ... --output-dir runs/feedback

# same proposal/evaluation budget, but no measured search feedback
python3 -m prompt_compiler.cli ... --output-dir runs/no_feedback --no-feedback
```

Use a convergence patience greater than the round count when the experiment must consume exactly the same round budget. Compare `search_archive.json`, `compression_report.json`, and `pareto_frontier.csv`.

## Artifacts

Each run writes:

- `best_prompt.txt`
- `compression_report.json`
- `search_archive.json`
- `pareto_frontier.csv`
- `dev_frontier.csv`
- `candidate_prompts.jsonl`
- `candidate_reports.jsonl`
- `candidate_outputs.jsonl`
- `reference_dataset.jsonl`
- `failures.json`
- `proposer_traces.jsonl`
- `run_events.jsonl`

`search_archive.json` is the compact learning record: complete prompts, parent relationships, actual instruction-token savings, behavior loss, and per-example residuals across rounds.
