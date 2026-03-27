import torch
import time
import subprocess
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def extract_assertions(test_code: str, entry_point: str):
    lines = [line.strip() for line in test_code.splitlines() if line.strip()]
    assertions = [line for line in lines if line.startswith("assert ")]
    if assertions:
        return [f"def check(candidate):\n    {assertion}\n\ncheck({entry_point})" for assertion in assertions]
    return [test_code + f"\ncheck({entry_point})"]


# --- 1. SANDBOX RUNNER ---
def run_in_sandbox(generated_code: str, test_code: str) -> bool:
    full_code = generated_code + "\n\n" + test_code
    try:
        result = subprocess.run(
            ["python3", "4_sandbox.py"],
            input=full_code, text=True, capture_output=True, timeout=5
        )
        return result.returncode == 0 and "SUCCESS" in result.stdout
    except Exception:
        return False


def score_candidate(generated_code: str, test_code: str, entry_point: str) -> float:
    assertion_tests = extract_assertions(test_code, entry_point)
    passed = 0

    for single_test in assertion_tests:
        if run_in_sandbox(generated_code, single_test):
            passed += 1

    return passed / len(assertion_tests)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # --- 2. SETUP & PROFILING PREP ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./lora-finetuned"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    dataset_split = os.getenv("TASK3_DATASET_SPLIT", "test[60:80]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)
    num_problems = len(dataset)

    # The N values we want to test per the instructions
    n_values_env = os.getenv("TASK3_N_VALUES")
    n_values = [int(value.strip()) for value in n_values_env.split(",")] if n_values_env else [5, 10, 20, 50]
    results = []

    print(f"\n=== RUNNING BEST-OF-N SCALING ANALYSIS ===")

    # --- 3. THE SCALING LOOP ---
    for n in n_values:
        print(f"\n--- Testing n = {n} ---")

        problems_solved = 0
        total_inference_time = 0.0
        total_test_time = 0.0
        best_samples_to_50 = None

        # Clear Mac GPU cache to get accurate VRAM readings
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        for i in range(len(dataset)):
            prompt = dataset[i]["prompt"]
            entry_point = dataset[i]["entry_point"]
            raw_test_code = dataset[i]["test"]
            test_code = raw_test_code + f"\ncheck({entry_point})"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # A. PROFILE INFERENCE LATENCY
            start_inf = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,  # Higher temp for diverse answers
                    num_return_sequences=n,
                    top_p=0.95
                )
            inf_time = time.time() - start_inf
            total_inference_time += inf_time

            # Check peak VRAM usage for this N
            vram_mb = 0
            if torch.backends.mps.is_available():
                vram_mb = torch.mps.current_allocated_memory() / (1024 ** 2)

            # B. PROFILE TESTING OVERHEAD (Sandbox)
            start_test = time.time()
            problem_passed = False
            best_score = -1.0

            for j in range(n):
                completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                generated_code = prompt + completion

                candidate_score = score_candidate(generated_code, raw_test_code, entry_point)
                best_score = max(best_score, candidate_score)

                if candidate_score == 1.0:
                    problem_passed = True
                    break

            test_time = time.time() - start_test
            total_test_time += test_time

            if problem_passed:
                problems_solved += 1

            running_accuracy = (problems_solved / (i + 1)) * 100
            if best_samples_to_50 is None and running_accuracy >= 50.0:
                best_samples_to_50 = (i + 1) * n

        # C. CALCULATE METRICS
        accuracy = (problems_solved / num_problems) * 100
        total_wall_clock = total_inference_time + total_test_time

        print(f"Accuracy: {accuracy:.1f}%")
        print(
            f"Total Wall-Clock: {total_wall_clock:.2f}s (Inference: {total_inference_time:.2f}s | Testing: {total_test_time:.2f}s)")
        print(f"Peak VRAM: {vram_mb:.1f} MB")
        if best_samples_to_50 is not None:
            print(f"Samples Needed to Reach 50% Running Accuracy: {best_samples_to_50}")
        else:
            print("Samples Needed to Reach 50% Running Accuracy: Not reached")

        results.append({
            "n": n,
            "accuracy": accuracy,
            "wall_clock": total_wall_clock,
            "inference_time": total_inference_time,
            "test_time": total_test_time,
            "vram_mb": vram_mb,
            "samples_to_50": best_samples_to_50
        })

    # --- 4. PRINT SUMMARY FOR PARETO CURVE ---
    print("\n" + "=" * 65)
    print("=== BEST-OF-N EFFICIENCY SUMMARY (Pareto Curve Data) ===")
    print("=" * 65)
    print(f"{'N':<5} | {'Accuracy':<10} | {'Total Time (s)':<16} | {'VRAM (MB)':<10} | {'Samples->50%':<12}")
    print("-" * 82)
    for r in results:
        samples_text = str(r['samples_to_50']) if r['samples_to_50'] is not None else "Not reached"
        print(f"{r['n']:<5} | {r['accuracy']:<9.1f}% | {r['wall_clock']:<15.2f} | {r['vram_mb']:<10.1f} | {samples_text:<12}")
    print("=" * 82)


if __name__ == "__main__":
    main()
