import torch
import subprocess
import os
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


STATUS_MARKER = "__SANDBOX_STATUS__:"


# ==========================================
# 1. SANDBOX EXECUTION FUNCTION
# ==========================================
def run_in_sandbox(generated_code: str, test_code: str) -> str:
    """Passes code to the sandbox.py process and classifies the outcome."""
    full_code = generated_code + "\n\n" + test_code

    try:
        # Spawn the sandbox process securely (Requires sandbox.py to be in the same folder)
        result = subprocess.run(
            ["python3", "4_sandbox.py"],
            input=full_code,
            text=True,
            capture_output=True,
            timeout=7  # 2 seconds longer than the internal 5s alarm, just in case
        )

        combined_output = f"{result.stdout}\n{result.stderr}"

        if result.returncode == 0:
            return "SUCCESS"
        elif f"{STATUS_MARKER}TIMEOUT_ERROR" in combined_output:
            return "TIMEOUT"
        elif f"{STATUS_MARKER}WRONG_OUTPUT_ERROR" in combined_output:
            return "WRONG_OUTPUT"
        elif f"{STATUS_MARKER}ASSERTION_ERROR" in combined_output:
            return "ASSERTION_ERROR"
        elif f"{STATUS_MARKER}EXCEPTION_ERROR" in combined_output:
            return "EXCEPTION"
        else:
            return "UNKNOWN_SYSTEM_ERROR"

    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # ==========================================
    # 2. LOAD YOUR FINE-TUNED MODEL
    # ==========================================
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./lora-finetuned"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)

    print(f"Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    dataset_split = os.getenv("TASK1_DATASET_SPLIT", "test[:50]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)

    # Parameters
    n_samples = int(os.getenv("TASK1_N_SAMPLES", "20"))
    max_tokens = int(os.getenv("TASK1_MAX_TOKENS", "512"))
    temperature = float(os.getenv("TASK1_TEMPERATURE", "0.4"))

    results_tally = []

    print(f"\n=== GENERATING & EVALUATING {len(dataset) * n_samples} FUNCTIONS ===")

    # ==========================================
    # 3. GENERATE AND CLASSIFY LOOP
    # ==========================================
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        entry_point = dataset[i]["entry_point"]
        test_code = dataset[i]["test"] + f"\ncheck({entry_point})"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Ask your model to generate 20 different solutions
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            top_p=0.95
        )

        print(f"Evaluating Problem {i + 1}/{len(dataset)}...")

        for j in range(n_samples):
            # Extract the AI's code
            completion = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_code = prompt + completion

            # Send it to the sandbox!
            outcome = run_in_sandbox(generated_code, test_code)
            results_tally.append(outcome)

    # ==========================================
    # 4. PRINT THE FINAL TAXONOMY
    # ==========================================
    counts = Counter(results_tally)
    total = len(results_tally)

    print("\n" + "=" * 40)
    print("=== FINAL SYSTEMATIC FAILURE ANALYSIS ===")
    print("=" * 40)
    print(f"Total Evaluated: {total}")
    print(f"Success Rate:    {(counts.get('SUCCESS', 0) / total) * 100:.2f}%\n")

    # <--- CHANGED: Updated the print labels to match the new strict requirements
    print("--- FAILURE BREAKDOWN ---")
    print(f"Timeouts (Infinite Loops):   {(counts.get('TIMEOUT', 0) / total) * 100:.2f}%")
    print(f"Exceptions (Crashes/Syntax): {(counts.get('EXCEPTION', 0) / total) * 100:.2f}%")
    print(f"Assertion Errors (AI code):  {(counts.get('ASSERTION_ERROR', 0) / total) * 100:.2f}%")
    print(f"Wrong Outputs (Failed Test): {(counts.get('WRONG_OUTPUT', 0) / total) * 100:.2f}%")
    print(f"Unknown System Errors:       {(counts.get('UNKNOWN_SYSTEM_ERROR', 0) / total) * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
