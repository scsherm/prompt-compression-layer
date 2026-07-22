# Feedback optimizer: scoped-investigation smoke test

Date: 2026-07-22

## Question

Can feedback-conditioned full-prompt compression reduce a SOC extraction contract while preserving ground-truth extraction on unseen reports containing a historical-incident distractor?

## Setup

- Dataset: 20 labeled rows from the public-derived `scoped_investigation` suite.
- Target: `gpt-5-nano-2025-08-07`.
- Proposer: `gpt-5.4-mini-2026-03-17`.
- Search: two rounds, two full-prompt proposals per round.
- Split: 12 search, 4 dev, 4 holdout.
- Task metrics: field/value precision, recall, F1, exact match, JSON validity, and schema validity.

The task requires extracting only the current incident while ignoring a realistic related historical incident.

## Selected prompt

- Original instruction tokens: 209.
- Compressed instruction tokens: 124.
- Reduction: 40.67%.
- Search behavior regression: 0.
- Dev precision/recall/F1: 1.0 / 1.0 / 1.0.

## Restriction-induced false failure

The first holdout pass was launched with an explicit 2,048 output-token ceiling. One candidate response ended with `status=incomplete`, `reason=max_output_tokens`, and an empty returned completion. That produced an artificial holdout F1 of 0.8571.

The client timeout and default output-token caps were removed after this diagnosis.

## Uncapped holdout rerun

Both original and compressed prompts were rerun on all four holdout rows with no request timeout, no output-token ceiling, and the same unspecified target reasoning setting used during optimization.

| metric | original | compressed |
|---|---:|---:|
| precision | 1.0 | 1.0 |
| recall | 1.0 | 1.0 |
| F1 | 1.0 | 1.0 |
| exact match rate | 1.0 | 1.0 |
| valid JSON rate | 1.0 | 1.0 |
| schema validity rate | 1.0 | 1.0 |

All eight uncapped target calls completed successfully. The compressed prompt preserved every labeled field on the four unseen examples while reducing reusable instruction tokens by 40.67%.

## Interpretation

This is a successful end-to-end smoke test, not a generalization claim. It verifies that the new feedback loop can learn from labeled extraction behavior and that the resulting prompt can survive an unseen distractor holdout. A defensible generality claim requires the full task-by-prompt-style matrix and a larger frozen holdout.
