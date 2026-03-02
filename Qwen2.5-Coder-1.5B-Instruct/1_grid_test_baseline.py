import torch
import evaluate
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    # 1. Setup & Safety
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model & Metric
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    code_eval = evaluate.load("code_eval")

    # Testing on 10 problems to keep the total run time manageable.
    # (Increase this to 40 later if you want a true benchmark)
    dataset = load_dataset("openai_humaneval", split="test[:10]")

    # 3. Define the Grid Search Parameters
    n_options = [5, 10, 20]
    temp_options = [0.2, 0.4, 0.8]
    token_options = [256, 512]

    # File to save our results so we don't lose data
    results_file = "grid_search_results.json"
    all_results = []

    experiment_num = 1
    total_experiments = len(n_options) * len(temp_options) * len(token_options)

    print(f"\n=== STARTING GRID SEARCH ({total_experiments} EXPERIMENTS) ===")

    # 4. The Grid Search Loops
    for max_tokens in token_options:
        for temperature in temp_options:
            for n_samples in n_options:

                print(f"\n--- Experiment {experiment_num}/{total_experiments} ---")
                print(f"Params: n={n_samples} | temp={temperature} | max_tokens={max_tokens}")

                predictions = []
                references = []

                # Evaluate all problems for this specific setup
                for i in range(len(dataset)):
                    prompt = dataset[i]["prompt"]
                    entry_point = dataset[i]["entry_point"]

                    test_code = dataset[i]["test"] + f"\ncheck({entry_point})"
                    references.append(test_code)

                    inputs = tokenizer(prompt + "\n    ", return_tensors="pt").to(device)

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
                        completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:],
                                                      skip_special_tokens=True)
                        full_code = prompt + "\n    " + completion
                        problem_predictions.append(full_code)

                    predictions.append(problem_predictions)

                # 5. Calculate pass@k dynamically based on n_samples
                # We can't ask for pass@10 if we only generated 5 samples
                k_values = [1, 5]
                if n_samples >= 10: k_values.append(10)
                if n_samples == 20: k_values.append(20)

                print("Running unit tests...")
                pass_at_k, _ = code_eval.compute(
                    references=references,
                    predictions=predictions,
                    k=k_values
                )

                # 6. Record and Save Results
                result_entry = {
                    "experiment": experiment_num,
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "scores": {f"pass@{k}": round(pass_at_k[f"pass@{k}"] * 100, 2) for k in k_values}
                }

                print(f"Results: {result_entry['scores']}")

                all_results.append(result_entry)

                # Save to file immediately so data is safe
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=4)

                experiment_num += 1

    print("\n=== GRID SEARCH COMPLETE ===")
    print(f"All results safely saved to {results_file}")


if __name__ == "__main__":
    main()