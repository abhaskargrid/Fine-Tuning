# Python Function Synthesis (HumanEval) - Fine-Tuning Project

This project fine-tunes code LLMs with LoRA to generate Python functions that pass HumanEval unit tests.

## Problem Statement
`19. Python Function Synthesis (HumanEval)`

- Write Python functions that pass unit tests.
- Use: `torch`, `transformers`, `peft`, `trl`, `datasets`, `evaluate`, `multiprocessing`.
- Mac setup: use `MPS` for generation/training, CPU-bound execution for tests.
- Train on manageable HumanEval subsets on Mac.
- Use best-of-n sampling.
- Goal: improve over baseline without overfitting to a continuous block.

## What Was Done

1. Started with a very tiny model (about `15MB`) to understand fine-tuning flow.
2. Tuned `HuggingFaceTB/SmolLM2-360M-Instruct` and verified tuning behavior.
3. Switched to code models:
   - `Qwen/Qwen2.5-Coder-0.5B`
   - `Qwen/Qwen2.5-Coder-1.5B-Instruct`
4. Ran baseline HumanEval experiments with parameter sweeps.
5. Applied LoRA fine-tuning experiments on HumanEval subsets.
6. Added an updated 1.5B experiment using non-contiguous training chunks to reduce overfitting risk.
7. Built a Streamlit app for prompt -> generation -> test-runner (+ MCTS-style search loop).

## Repository Structure

- `Qwen2.5-Coder-0.5B/`
  - `1_eval_baseline.py`
  - `1_grid_test_baseline.py`
  - `2_finetune_lora.py`
  - `3_eval_lora.py`
  - `4_eval_lora_top.py`
  - `app.py`
  - `grid_search_results.json`
- `Qwen2.5-Coder-1.5B-Instruct/`
  - `1_eval_baseline.py`
  - `2_finetune_lora.py`
  - `3_eval_lora.py`
  - `app.py`
  - `output.json` (run logs/results)
- `Qwen2.5-Coder-1.5B-Instruct-update/`
  - `1_eval_baseline.py`
  - `2_finetune_lora.py`
  - `3_eval_lora.py`
  - `app.py`
  - `grid_search_results.json`
  - `output.json`
- `test.txt`
  - Example prompt + test format for manual verification.

## Environment Setup (Mac)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft trl datasets evaluate streamlit accelerate
```

Optional:

```bash
export TOKENIZERS_PARALLELISM=false
```

## Step-by-Step Workflow

### Step 1: Baseline evaluation

Run baseline before fine-tuning.

For `0.5B`:

```bash
cd Qwen2.5-Coder-0.5B
python3 1_eval_baseline.py
```

For `1.5B-Instruct`:

```bash
cd Qwen2.5-Coder-1.5B-Instruct
python3 1_eval_baseline.py
```

For `1.5B-Instruct-update`:

```bash
cd Qwen2.5-Coder-1.5B-Instruct-update
python3 1_eval_baseline.py
```

Current eval slice in update folder baseline script:

```python
dataset = load_dataset("openai_humaneval", split="test[130:135]")
```

### Step 2: (Optional) Baseline grid search

This tests combinations:
- `n_samples`: `5, 10, 20`
- `temperature`: `0.2, 0.4, 0.8`
- `max_tokens`: `256, 512`

For `0.5B`:

```bash
cd Qwen2.5-Coder-0.5B
python3 1_grid_test_baseline.py
```

Result file: `grid_search_results.json`

### Step 3: LoRA fine-tuning

For `1.5B-Instruct-update`, fine-tuning uses non-contiguous HumanEval chunks (not continuous ranges like `0:40`):

```python
dataset = load_dataset(
    "openai_humaneval",
    split="test[30:40]+test[60:70]+test[90:100]+test[120:130]+test[150:160]"
)
```

This setup is used specifically for tuning in `Qwen2.5-Coder-1.5B-Instruct-update/2_finetune_lora.py`.

Run tuning:

```bash
cd Qwen2.5-Coder-1.5B-Instruct-update
python3 2_finetune_lora.py
```

Adapter save path:
- `Qwen2.5-Coder-1.5B-Instruct-update/lora-finetuned/`

### Step 4: Evaluate tuned model

For `1.5B-Instruct-update`:

```bash
cd Qwen2.5-Coder-1.5B-Instruct-update
python3 3_eval_lora.py
```

Current eval slice in update folder post-tuning script:

```python
dataset = load_dataset("openai_humaneval", split="test[130:135]")
```

This means training and evaluation slices are different in this setup.

### Step 5: Run demo app (prompt -> code -> test)

For `0.5B`:

```bash
cd Qwen2.5-Coder-0.5B
streamlit run app.py
```

For `1.5B-Instruct`:

```bash
cd Qwen2.5-Coder-1.5B-Instruct
streamlit run app.py
```

For `1.5B-Instruct-update`:

```bash
cd Qwen2.5-Coder-1.5B-Instruct-update
streamlit run app.py
```

## How to Run Code in Different Folders

From repo root:

```bash
cd Qwen2.5-Coder-0.5B
# run scripts here...

cd ../Qwen2.5-Coder-1.5B-Instruct
# run scripts here...

cd ../Qwen2.5-Coder-1.5B-Instruct-update
# run scripts here...
```

Each folder is self-contained and expects relative paths to its own LoRA adapter directory.

## Reported Results

### Previous Results (Earlier Folders)

- `Qwen2.5-Coder-0.5B`
  - Baseline (`n=5, temp=0.4, max_tokens=256`): Pass@1 `26%`, Pass@5 `60%`
  - Another baseline (`n=5, temp=0.8, max_tokens=256`): Pass@1 `16%`, Pass@5 `40%`
  - Post-LoRA on `40` problems (same params): Pass@1 `18%`, Pass@5 `80%`

- `Qwen2.5-Coder-1.5B-Instruct`
  - Baseline (`n=5, temp=0.8, max_tokens=256`): Pass@1 `16%`, Pass@5 `50%`
  - Early post-LoRA: Pass@1 `16%`, Pass@5 `40%`
  - After hyperparameter/epoch changes: Pass@1 `74%`, Pass@5 `90%`
  - Unseen slices examples:
    - Next 10: Pass@1 `56%`, Pass@5 `90%`
    - Another 10: Pass@1 `66%`, Pass@5 `90%`

### Latest Results (`Qwen2.5-Coder-1.5B-Instruct-update`)

For eval on `test[130:135]`:

- Baseline (`python3 1_eval_baseline.py`)
  - Pass@1: `0.00%`
  - Pass@5: `0.00%`
- Post-LoRA (`python3 3_eval_lora.py`)
  - Pass@1: `20.00%`
  - Pass@5: `20.00%`

Interpretation: this latest run shows improvement on the held evaluation slice while using chunked (non-contiguous) tuning data.

## Prompt and Test Format

Use `test.txt` format:

1. Write a function prompt with signature/docstring.
2. Write a `check(candidate)` test function with asserts.
3. Call `check(your_function_name)` at the end.

## Safety Notes

- Evaluation with `evaluate.load("code_eval")` executes generated code.
- Scripts set `HF_ALLOW_CODE_EVAL=1`.
- Keep the `if __name__ == "__main__":` guard for macOS multiprocessing safety.
- Streamlit app uses a timeout-based subprocess runner; keep this pattern for sandbox-style safety.

## Recommended Final Checks Before Submission

1. Keep training and evaluation slices disjoint (already done in the update folder).
2. Report average over multiple runs (different seeds) for stability.
3. Keep one fixed evaluation config for fair comparison (same `n`, `temperature`, `max_tokens`).
4. Include both chunked-train held-slice metrics and additional unseen-slice metrics in final report.
