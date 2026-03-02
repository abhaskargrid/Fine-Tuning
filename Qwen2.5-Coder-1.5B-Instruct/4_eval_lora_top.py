import torch
import evaluate
import os
import json
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
    dataset = load_dataset("openai_humaneval", split="test[:10]")

    # 3. The Top 4 Setups from your Baseline Grid Search
    top_setups = [
        {"name": "Exp 14 (Base Winner)", "n": 10, "temperature": 0.4, "max_tokens": 512},
        {"name": "Exp 5 (High Pass@5)", "n": 10, "temperature": 0.4, "max_tokens": 256},
        {"name": "Exp 17 (Creative)", "n": 10, "temperature": 0.8, "max_tokens": 512},
        {"name": "Exp 1 (Strict Logic)", "n": 5, "temperature": 0.2, "max_tokens": 256}
    ]

    all_results = []

    print(f"\n=== TESTING TOP {len(top_setups)} CONFIGURATIONS ===")

    # 4. Loop through only the best setups
    for setup in top_setups:
        print(f"\n--- Running {setup['name']} ---")
        print(f"Params: n={setup['n']} | temp={setup['temperature']} | max_tokens={setup['max_tokens']}")

        predictions = []
        references = []

        for i in range(len(dataset)):
            prompt = dataset[i]["prompt"]
            entry_point = dataset[i]["entry_point"]

            test_code = dataset[i]["test"] + f"\ncheck({entry_point})"
            references.append(test_code)

            # No indent trick here, LoRA handles it now
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=setup['max_tokens'],
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=setup['temperature'],
                num_return_sequences=setup['n'],
                top_p=0.95
            )

            problem_predictions = []
            for j in range(setup['n']):
                completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                full_code = prompt + completion
                problem_predictions.append(full_code)

            predictions.append(problem_predictions)

        # Set the k-values dynamically based on n
        k_values = [1, 5]
        if setup['n'] >= 10: k_values.append(10)

        print("Calculating metrics...")
        pass_at_k, _ = code_eval.compute(
            references=references,
            predictions=predictions,
            k=k_values
        )

        # Print and save results
        print(f"Results for {setup['name']}:")
        scores = {f"pass@{k}": round(pass_at_k[f"pass@{k}"] * 100, 2) for k in k_values}
        for k, v in scores.items():
            print(f"  {k}: {v}%")

        all_results.append({
            "setup_name": setup['name'],
            "scores": scores
        })

    print("\n=== FINAL COMPARISON ===")
    for res in all_results:
        print(f"{res['setup_name']}: {res['scores']}")


if __name__ == "__main__":
    main()