# Behavioral Prompt Compression Architecture

The compiler searches for shorter prompt templates that preserve target-model behavior:

```text
given:  target model M, original prompt template P, inputs X
return: compressed prompt template P'
such that M(P', x) behaves like M(P, x) for many x in X
```

Reference outputs are behavioral references, not objective truth labels.

## Layer Diagram

```mermaid
flowchart TD
    A[Original prompt template P] --> B[1. Chunking layer]
    B --> C[Prompt chunks: role, task, constraints, schema, examples, input slot]
    C --> D[2. Exploration layer]
    D --> E[Candidate prompts P']
    E --> F[3. Input variance and evaluation layer]
    X[Input set X] --> R[Reference dataset: M(P, x)]
    R --> F
    F --> G[Candidate reports]
    G --> H[4. Optimizer epoch loop]
    H --> I[Pareto frontier]
    I --> J[Credit assignment]
    J --> K[Mutation and recombination]
    K --> D
    I --> L[Best prompt and artifacts]
```

## Layers

### 1. Chunking Layer

The chunking layer breaks the invariant instruction layer of `P` into units that can be rewritten independently. The input placeholder is protected in the first version.

Chunk types include:

- role
- task
- constraint
- negative constraint
- output schema
- example
- style
- input slot
- safety
- tool instruction

### 2. Exploration Layer

The exploration layer applies rewrite operators to chunks and assembles candidate prompt templates.

Current operator families include:

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

The tokenizer is part of this layer. Candidate text is measured under the configured tokenizer, so Mandarin, DSL, and mixed forms are evaluated empirically.

### 3. Input Variance and Evaluation Layer

The evaluation layer runs each candidate across varied inputs and compares candidate outputs to reference outputs.

Metrics include:

- instruction-token reduction
- Euclidean embedding drift
- deterministic contract checks
- JSON/schema validity
- task-specific checks
- output-language drift
- optional LLM judge disagreement
- repeated-run variance

Euclidean embedding distance is one metric. It is not the optimizer by itself.

### 4. Optimizer Epoch Loop

Each epoch evaluates candidates, keeps the Pareto frontier, assigns credit to chunk/operator choices, and creates the next population.

```text
chunk -> explore -> evaluate across inputs -> Pareto select
      -> credit assignment -> mutate/recombine -> next epoch
```

The optimizer does not select only the lowest-loss prompt early. It preserves trade-offs between compression and behavior preservation.

