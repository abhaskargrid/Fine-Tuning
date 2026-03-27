import torch
import re
import subprocess
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. ABLATION FUNCTIONS
# ==========================================
def remove_docstring(prompt: str) -> str:
    no_docstring = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', prompt)
    no_docstring = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', no_docstring)
    import os
    return os.linesep.join([s for s in no_docstring.splitlines() if s.strip()])


def remove_type_hints(prompt: str) -> str:
    cleaned = re.sub(r'->\s*[^:]+:', ':', prompt)
    cleaned = re.sub(r':\s*[A-Za-z_][A-Za-z0-9_\[\]]*\s*(?=[,\)])', '', cleaned)
    return cleaned


# ==========================================
# 2. SANDBOX EVALUATOR
# ==========================================
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


# ==========================================
# 3. GENERATION & SCORING LOOP
# ==========================================
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Host Device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    dataset_split = os.getenv("TASK4_DATASET_SPLIT", "test[:20]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)

    scores = {"Original": 0, "No_Types": 0, "No_Docstring": 0}

    print("\n" + "=" * 50)
    print("🔬 RUNNING ABLATION ACCURACY EVALUATION")
    print("=" * 50)

    for i in range(len(dataset)):
        original_prompt = dataset[i]["prompt"]
        entry_point = dataset[i]["entry_point"]
        test_code = dataset[i]["test"] + f"\ncheck({entry_point})"

        # Create our blindfolded prompts
        no_types_prompt = remove_type_hints(original_prompt)
        no_docs_prompt = remove_docstring(original_prompt)

        prompts_to_test = {
            "Original": original_prompt,
            "No_Types": no_types_prompt,
            "No_Docstring": no_docs_prompt
        }

        print(f"\nEvaluating Problem {i + 1}/20...")

        for condition, prompt_text in prompts_to_test.items():
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id,
                                         temperature=0.2)

            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_code = prompt_text + completion

            # Evaluate!
            if run_in_sandbox(generated_code, test_code):
                scores[condition] += 1
                print(f"  [{condition}] ✅ Passed")
            else:
                print(f"  [{condition}] ❌ Failed")

    # ==========================================
    # 4. PRINT THE CODE READABILITY METRIC
    # ==========================================
    total = len(dataset)
    print("\n" + "=" * 50)
    print("📊 FINAL ABLATION RESULTS (Code Readability Metric)")
    print("=" * 50)
    print(f"Original Prompt Accuracy:     {(scores['Original'] / total) * 100:.1f}%")
    print(f"No Type Hints Accuracy:       {(scores['No_Types'] / total) * 100:.1f}%")
    print(f"No Docstring Accuracy:        {(scores['No_Docstring'] / total) * 100:.1f}%")

    drop_types = ((scores['Original'] - scores['No_Types']) / total) * 100
    drop_docs = ((scores['Original'] - scores['No_Docstring']) / total) * 100

    print("\n--- ACCURACY DROP ---")
    print(f"Removing Types caused a   {drop_types:+.1f}% drop in accuracy.")
    print(f"Removing Docs caused a    {drop_docs:+.1f}% drop in accuracy.")
    print("=" * 50)


if __name__ == "__main__":
    main()
