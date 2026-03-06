# Python Function Synthesis (HumanEval) - Qwen2.5-Coder-1.5B-Instruct

## Full Report

Report link: https://docs.google.com/document/d/1kILVV77s5awhAQEhe9enq44jEpyCO8fPcUy_BBwj48k/edit?tab=t.0#heading=h.9ln0wkhzf24p

## Problem Statement
`19. Python Function Synthesis (HumanEval)`

- Write Python functions that pass unit tests.
- Libraries: `torch`, `transformers`, `peft`, `trl`, `datasets`, `evaluate`, `multiprocessing`.
- Mac acceleration: use `MPS` for generation/training, CPU for test execution.
- Focus on 30-50 HumanEval problems on Mac.
- Use best-of-n sampling.
- Goal: `5+` point improvement over SFT baseline.
- Data: `openai/openai_humaneval`.

## Folder Covered in This README

This README documents only:
- `Qwen2.5-Coder-1.5B-Instruct/`

Main files:
- `1_eval_baseline.py`: baseline evaluation.
- `2_finetune_lora.py`: LoRA fine-tuning.
- `3_eval_lora.py`: evaluate tuned model.
- `app.py`: Streamlit simulation (prompt -> code -> test, with MCTS-style search).
- `output.json`: collected logs and results.

## Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft trl datasets evaluate streamlit accelerate
```

## How the 1.5B Folder Works

### 1) Run baseline test

```bash
cd Qwen2.5-Coder-1.5B-Instruct
python3 1_eval_baseline.py
```

What it does:
- Loads base model `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- Uses HumanEval test split.
- Generates `n_samples` candidates and computes `pass@k` with `evaluate/code_eval`.

### 2) Run LoRA fine-tuning

```bash
cd Qwen2.5-Coder-1.5B-Instruct
python3 2_finetune_lora.py
```

What it does:
- Loads HumanEval subset (`test[:40]` in current script).
- Applies LoRA (`q_proj`, `v_proj`) with PEFT.
- Trains and saves adapter into `./lora-finetuned`.

### 3) Test tuned model

```bash
cd Qwen2.5-Coder-1.5B-Instruct
python3 3_eval_lora.py
```

What it does:
- Loads base model + `./lora-finetuned` adapter.
- Generates multiple candidates per task.
- Runs HumanEval tests and prints `Pass@1`, `Pass@5`.

Note:
- You can change the evaluation slice inside `3_eval_lora.py` to test reliability on unseen ranges, for example:
  - `test[:10]`
  - `test[120:130]`
  - any other holdout split.

### 4) Run Streamlit simulation

```bash
cd Qwen2.5-Coder-1.5B-Instruct
streamlit run app.py
```

What it does:
- UI for prompt + tests input. (use the prompt and test from the test.txt file or give your own test and promt in the same format)
- Generates code with tuned model.
- Executes tests in a subprocess with timeout.
- Uses search iterations (MCTS-style loop) to try multiple branches.

## Typical End-to-End Run Order

```bash
cd Qwen2.5-Coder-1.5B-Instruct
python3 1_eval_baseline.py
python3 2_finetune_lora.py
python3 3_eval_lora.py
streamlit run app.py
```

## Safety Notes

- `HF_ALLOW_CODE_EVAL=1` is required for HumanEval metrics.
- Generated code execution is potentially unsafe; keep sandboxed execution and timeouts.
- Keep `if __name__ == "__main__":` guards for macOS multiprocessing safety.
