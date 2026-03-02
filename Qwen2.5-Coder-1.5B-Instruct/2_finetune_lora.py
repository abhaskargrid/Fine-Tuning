import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
output_dir = "./lora-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# LoRA Setup
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("openai_humaneval", split="test[:40]")


def tokenize_function(examples):
    texts = []

    for prompt, solution in zip(examples["prompt"], examples["canonical_solution"]):

        # --- DATA PREP EXPERIMENT: SIGNATURE ONLY ---
        # HumanEval prompts look like: def name(args):\n    """docstring"""\n
        # We can split the string at the quotes to throw away the English docstring.

        if '"""' in prompt:
            # Keep only the signature part (before the docstring)
            signature_only = prompt.split('"""')[0]
        else:
            signature_only = prompt

        # Now we glue the cleaned signature to the solution
        final_text = signature_only + solution + tokenizer.eos_token
        texts.append(final_text)

    # Translate the cleaned text into numbers for LoRA
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=15,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n--- STARTING LoRA FINE-TUNING ---")
trainer.train()

print(f"\nSaving LoRA adapter to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
