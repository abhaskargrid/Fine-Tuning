---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- code-generation
- qwen2.5-coder
- humaneval
---

# Qwen2.5-Coder-1.5B-Instruct LoRA Adapter

This repository contains a LoRA adapter fine-tuned on HumanEval-style Python function synthesis tasks.

## Base Model

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Adapter type: LoRA
- Task type: causal language modeling / code generation

## Included Files

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

## How To Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_repo = "abhi0619/qwen25-coder-15b-instruct-lora-humaneval"

tokenizer = AutoTokenizer.from_pretrained(adapter_repo)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_repo)
```

## Notes

- This is an adapter repository, not a merged full-weight model.
- You must load it together with the base Qwen model.
- The tokenizer files are included for convenience.
