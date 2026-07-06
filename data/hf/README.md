# Hugging Face Instruction Data

This directory contains normalized instruction-data samples for prompt compression experiments.

Current sample:

```text
data/hf/no_robots_100.jsonl
```

Source dataset:

```text
HuggingFaceH4/no_robots
```

Normalized row fields:

- `id`
- `input`
- `reference_output`
- `dataset`
- `category`
- `source_row`

`input` is the instruction text used as the variable prompt input. `reference_output` is the dataset completion supplied by the source dataset.

Refresh the sample:

```bash
/Users/scsherm/anaconda3/envs/prompt-compression-layer/bin/python \
  scripts/download_instruction_dataset.py \
  --dataset HuggingFaceH4/no_robots \
  --config default \
  --split train \
  --limit 100 \
  --output data/hf/no_robots_100.jsonl
```

The source dataset license is `cc-by-nc-4.0`.
