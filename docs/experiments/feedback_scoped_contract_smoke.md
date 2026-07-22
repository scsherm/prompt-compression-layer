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

The extraction contract scopes outputs to the current incident in reports that also contain a realistic related historical incident.

## Selected prompt

- Original instruction tokens: 209.
- Compressed instruction tokens: 124.
- Reduction: 40.67%.
- Search behavior regression: 0.
- Dev precision/recall/F1: 1.0 / 1.0 / 1.0.

## Holdout results

The original and compressed prompts were evaluated on all four holdout rows.

| metric | original | compressed |
|---|---:|---:|
| precision | 1.0 | 1.0 |
| recall | 1.0 | 1.0 |
| F1 | 1.0 | 1.0 |
| exact match rate | 1.0 | 1.0 |
| valid JSON rate | 1.0 | 1.0 |
| schema validity rate | 1.0 | 1.0 |

The compressed prompt preserved every labeled field on the four unseen examples while reducing reusable instruction tokens by 40.67%.

## Interpretation

This end-to-end smoke test validates learning from labeled extraction behavior and evaluation on an unseen distractor holdout. Broader generalization can be measured with a task-by-prompt-style matrix and a larger frozen holdout.
