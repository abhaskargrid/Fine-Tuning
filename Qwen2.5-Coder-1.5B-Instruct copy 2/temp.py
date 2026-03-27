# OLD Sandbox

import sys
import signal
import resource
import traceback

def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded the 5-second hard limit.")


def set_limits():
    # Set Memory Limit to 1GB (1024 * 1024 * 1024 bytes)
    gigabyte = 1024 * 1024 * 1024

    # resource.setrlimit(resource.RLIMIT_AS, (gigabyte, gigabyte))

    # Leave the CPU Timeout active:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

if __name__ == "__main__":
    # Read the AI-generated code passed from the main controller
    code_to_run = sys.stdin.read()

    try:
        set_limits()
        # Execute the untrusted code in a blank global dictionary
        exec(code_to_run, {})
        print("SUCCESS")

    except TimeoutError as e:
        sys.stderr.write("TIMEOUT_ERROR\n")
        sys.exit(1)
    except AssertionError as e:
        sys.stderr.write("ASSERTION_ERROR\n")
        traceback.print_exc()
        sys.exit(2)
    except Exception as e:
        sys.stderr.write("EXCEPTION_ERROR\n")
        traceback.print_exc()
        sys.exit(3)


# OLD 6_full_taxonomy.py

import torch
import subprocess
import os
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


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

        if result.returncode == 0:
            return "SUCCESS"
        elif "TIMEOUT_ERROR" in result.stderr:
            return "TIMEOUT"
        elif "ASSERTION_ERROR" in result.stderr:
            return "ASSERTION_ERROR"
        elif "EXCEPTION_ERROR" in result.stderr:
            return "EXCEPTION"
        else:
            return "WRONG_OUTPUT"  # Did not crash, but failed tests silently

    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def main():
    # ==========================================
    # 2. LOAD YOUR FINE-TUNED MODEL
    # ==========================================
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./lora-finetuned"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    print(f"Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    # Load 50 problems. We will generate 20 answers for each = 1,000 total.
    dataset = load_dataset("openai_humaneval", split="test[:50]")

    # Parameters
    n_samples = 20
    max_tokens = 512
    temperature = 0.4

    results_tally = []

    print(f"\n=== GENERATING & EVALUATING 1,000 FUNCTIONS ===")

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

        print(f"Evaluating Problem {i + 1}/50...")

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

    print("--- FAILURE BREAKDOWN ---")
    print(f"Timeouts (Infinite Loops):   {(counts.get('TIMEOUT', 0) / total) * 100:.2f}%")
    print(f"Exceptions (Crashes/Syntax): {(counts.get('EXCEPTION', 0) / total) * 100:.2f}%")
    print(f"Assertion Errors (Logic):    {(counts.get('ASSERTION_ERROR', 0) / total) * 100:.2f}%")
    print(f"Wrong Outputs (Edge Cases):  {(counts.get('WRONG_OUTPUT', 0) / total) * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()