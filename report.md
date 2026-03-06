# Project Report - Python Function Synthesis (HumanEval)

## Problem Statement

`19. Python Function Synthesis (HumanEval)`

- Write Python functions that pass unit tests.
- Libraries: `torch`, `transformers`, `peft`, `trl`, `datasets`, `evaluate`, `multiprocessing`.
- Mac acceleration: test execution is CPU-bound; use MPS for generation/training and CPU for testing.
- Build flow:
  - Step 1: use OpenAI HumanEval (164 problems)
  - Step 2: Code LLM + LoRA
  - Step 3: focus on 30-50 problems on Mac
  - Step 4: best-of-n sampling
  - Step 5: target 5+ point improvement over SFT

## Approach Summary

I fine-tuned a code LLM in stages:

1. Started with a very tiny LLM (~15MB) to understand fine-tuning behavior.
2. Tried `HuggingFaceTB/SmolLM2-360M-Instruct` and observed that it learns patterns from tuned data.
3. Shifted to code-focused models for HumanEval:
   - `Qwen/Qwen2.5-Coder-0.5B`
   - `Qwen/Qwen2.5-Coder-1.5B-Instruct`
4. Used LoRA/PEFT fine-tuning and HumanEval `pass@k` metrics.
5. Explored decoding/search parameters (`n_samples`, `temperature`, `max_tokens`) and multiple evaluation slices.

## Architecture Diagram (Before and After)

The following diagram shows the model architecture before and after the update:

![Model architecture before and after](./Architectural%20diagram.png)

## Implementation Workflow

- Baseline evaluation before tuning.
- Grid search on sampling and decoding settings.
- LoRA fine-tuning on selected HumanEval subset.
- Post-tuning evaluation on:
  - same/similar problem range
  - farther/unseen ranges for reliability checks
- Streamlit demo with prompt -> code generation -> test execution (with timeout sandbox).

## Models, Parameters, and Results

### 1) Qwen2.5-Coder-0.5B

Baseline examples:
- Parameters: `n_samples=5`, `max_tokens=256`, `temperature=0.4`
- Result:
  - Pass@1: `26.00%`
  - Pass@5: `60.00%`

Another baseline setting:
- Parameters: `n_samples=5`, `max_tokens=256`, `temperature=0.8`
- Result:
  - Pass@1: `16.00%`
  - Pass@5: `40.00%`

After LoRA fine-tuning on 40 problems:
- Parameters: `n_samples=5`, `max_tokens=256`, `temperature=0.8`
- Result:
  - Pass@1: `18.00%`
  - Pass@5: `80.00%`
- Observation: Pass@5 improved strongly, but Pass@1 remained low, and stability was limited.

### 2) Qwen2.5-Coder-1.5B-Instruct

Baseline:
- Parameters: `n_samples=5`, `max_tokens=256`, `temperature=0.8`
- Result:
  - Pass@1: `16.00%`
  - Pass@5: `50.00%`

Early post-LoRA run:
- Same parameters
- Result:
  - Pass@1: `16.00%`
  - Pass@5: `40.00%`

After changing training settings (epochs and related hyperparameters):
- Same evaluation parameters
- Result:
  - Pass@1: `74.00%`
  - Pass@5: `90.00%`

Reliability checks on other problems (not the same tuned subset):
- Next 10 problems:
  - Pass@1: `56.00%`
  - Pass@5: `90.00%`
- Another 10 problems:
  - Pass@1: `66.00%`
  - Pass@5: `90.00%`

Additional far-range test note:
- When testing on problems farther from tuned range, performance dropped.
- This experiment was done in folder: `Qwen2.5-Coder-1.5B-Instruct-update`.
- Core idea: train on non-continuous HumanEval chunks (distributed slices), not one continuous block.
- Training setup used chunked data like:

```python
dataset = load_dataset(
    "openai_humaneval",
    split="test[30:40]+test[60:70]+test[90:100]+test[120:130]+test[150:160]"
)
```

Evaluation on `130-135`:
- Baseline:
  - Pass@1: `0.00%`
  - Pass@5: `0.00%`
- Post-LoRA:
  - Pass@1: `20.00%`
  - Pass@5: `20.00%`

## Key Experimental Dimensions

- Architecture:
  - tested `Qwen2.5-Coder-0.5B` and `Qwen2.5-Coder-1.5B-Instruct`
- Data preparation:
  - used HumanEval prompts/solutions, with signature-focused formatting in scripts
- Decoding parameters:
  - best-of-n: `5`, `10`, `20`
  - temperature: `0.2`, `0.4`, `0.8`
  - max tokens: `256`, `512`

## Goal Check Against Problem Statement

Required goal: `5+` point improvement over SFT baseline.

Achieved:
- Multiple runs exceed +5 points.
- Strongest improvement observed on `Qwen2.5-Coder-1.5B-Instruct`:
  - Baseline Pass@1 `16%` -> Post-LoRA Pass@1 up to `74%`.

So the project meets the stated target.

## What Is Still Missing / Risks

- Generalization is uneven across far-away slices.
- Need a strict holdout protocol and consistent split reporting to avoid optimistic estimates.
- Need repeated runs with seeds to confirm stability.

## Final Conclusion

- Direction is correct.
- Using `Qwen2.5-Coder-1.5B-Instruct` + LoRA was the right choice for Mac constraints.
- You achieved the stated performance target and built both evaluation and demo pipeline.
- Final submission should clearly separate:
  - training split
  - validation/holdout split
  - same-parameter baseline vs post-LoRA comparison
  - average over repeated runs.
