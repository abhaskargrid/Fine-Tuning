import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_id = "Qwen/Qwen2.5-Coder-0.5B"
adapter_dir = "./qwen-lora-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

dataset = load_dataset("openai_humaneval", split="test[:1]")

prompt = dataset[0]["prompt"]
# Notice: We are NOT adding the "\n    " indent trick here!
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n--- RAW MODEL OUTPUT FOR PROBLEM 1 ---")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False # Greedy decoding for a clean look
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
print("--------------------------------------")