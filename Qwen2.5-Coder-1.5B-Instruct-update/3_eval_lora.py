import torch
import evaluate
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


def main():
    # 1. Setup & Safety
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Base Model + LoRA Adapter
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./lora-finetuned"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    print(f"Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    code_eval = evaluate.load("code_eval")
    # dataset = load_dataset("openai_humaneval", split="test[:10]")  # Testing first 10 problems
    dataset = load_dataset("openai_humaneval", split="test[130:135]")
    # 3. Evaluation Parameters (From your winning Grid Search: Experiment 14)
    # n_samples = 10
    # max_tokens = 512
    # temperature = 0.4

    n_samples = 5  # Best-of-5 sampling
    max_tokens = 256
    temperature = 0.8

    predictions = []
    references = []

    print(f"\n--- RUNNING POST-TUNING EVALUATION (BEST-OF-{n_samples}) ---")
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        entry_point = dataset[i]["entry_point"]

        # Prepare the background unit test
        test_code = dataset[i]["test"] + f"\ncheck({entry_point})"
        references.append(test_code)

        # NOTE: Removed the indent trick! We pass the raw prompt because your LoRA model
        # is now smart enough to handle its own indentation.
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate code
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            top_p=0.95
        )

        problem_predictions = []
        for j in range(n_samples):
            # Decode the generated text (ignoring the prompt part)
            completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Combine raw prompt and completion
            full_code = prompt + completion

            problem_predictions.append(full_code)

        predictions.append(problem_predictions)
        print(f"Evaluated Problem {i + 1}/{len(dataset)}...")

    # 4. Calculate pass@k
    print("\nCalculating metrics (running unit tests in the background)...")
    pass_at_k, results = code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1, 5]  # Removed 10
    )

    # And remove this line:
    # print(f"Pass@10: {pass_at_k['pass@10'] * 100:.2f}%")

    print("\n=== POST-TUNING RESULTS ===")
    print(f"Pass@1:  {pass_at_k['pass@1'] * 100:.2f}%")
    print(f"Pass@5:  {pass_at_k['pass@5'] * 100:.2f}%")


# This safely launches the code on macOS without the endless loop crash!
if __name__ == "__main__":
    main()