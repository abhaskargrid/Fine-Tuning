import torch
import evaluate
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    # 1. Setup & Safety
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # Required to run code execution
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load Model & Metric
    # Change this line:
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    code_eval = evaluate.load("code_eval")
    dataset = load_dataset("openai_humaneval", split="test[130:135]")

    # 3. Evaluation Parameters
    n_samples = 5  # Best-of-5 sampling
    max_tokens = 256
    temperature = 0.8

    predictions = []
    references = []

    print(f"\n--- RUNNING BASELINE EVALUATION (BEST-OF-{n_samples}) ---")
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        entry_point = dataset[i]["entry_point"]

        # HumanEval tests contain a 'check' function. We must call it at the end to run the test.
        test_code = dataset[i]["test"] + f"\ncheck({entry_point})"
        references.append(test_code)

        inputs = tokenizer(prompt + "\n    ", return_tensors="pt").to(device)

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
            # Extract the completion
            completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Combine prompt + completion so the test has the full function to run
            full_code = prompt + "\n    " + completion
            problem_predictions.append(full_code)

        predictions.append(problem_predictions)
        print(f"Evaluated Problem {i + 1}/{len(dataset)}...")

    # 4. Calculate pass@k
    print("\nCalculating metrics (running unit tests)...")
    pass_at_k, results = code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1, 5]
    )

    print("\n=== BASELINE RESULTS ===")
    print(f"Pass@1 (Accuracy if we only check 1st answer): {pass_at_k['pass@1'] * 100:.2f}%")
    print(f"Pass@5 (Accuracy if we check all 5 answers):  {pass_at_k['pass@5'] * 100:.2f}%")


# This is the crucial fix for macOS multiprocessing!
if __name__ == "__main__":
    main()