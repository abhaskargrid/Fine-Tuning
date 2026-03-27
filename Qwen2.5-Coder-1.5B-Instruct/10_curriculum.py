import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ==========================================
# 1. CORE EXECUTION & REWARD LOGIC
# ==========================================
def run_single_test(full_code: str) -> bool:
    """Runs a single assert statement securely."""
    try:
        exec(full_code, {})
        return True
    except AssertionError:
        return False
    except Exception:
        # Fails on syntax errors or NameErrors
        return False


def evaluate_problem(code_str: str, test_string: str, entry_point: str):
    """Splits HumanEval tests, calculates reward, and finds the exact failing edge cases."""
    raw_tests = test_string.split('\n')
    assert_tests = [t.strip() for t in raw_tests if 'assert ' in t]

    if not assert_tests:
        assert_tests = [f"{test_string}\ncheck({entry_point})"]

    reward = 0.0
    failed_cases = []

    for test in assert_tests:
        # --- THE MAGIC FIX ---
        # We explicitly bind the word 'candidate' to the actual function name!
        executable_test = f"{code_str}\ncandidate = {entry_point}\n{test}"

        if run_single_test(executable_test):
            reward += 1.0
        else:
            reward -= 0.25
            failed_cases.append(test)

    # Normalize reward to a percentage-like score for easy sorting
    max_possible_reward = len(assert_tests) * 1.0
    normalized_score = (reward / max_possible_reward) * 100 if max_possible_reward > 0 else 0

    return normalized_score, failed_cases


# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_dir = "./lora-finetuned"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    dataset_split = os.getenv("TASK2_DATASET_SPLIT", "test[50:65]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)

    curriculum_data = []
    adversarial_bank = []

    print(f"\n=== GENERATING CODE & CALCULATING GRANULAR REWARDS ===")

    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        entry_point = dataset[i]["entry_point"]
        test_code = dataset[i]["test"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate 1 candidate per problem (Zero-Shot) to test baseline knowledge
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, temperature=0.2
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_code = prompt + completion

        # Evaluate the code with granular rewards
        score, failed_tests = evaluate_problem(generated_code, test_code, entry_point)

        curriculum_data.append({
            "problem_id": dataset[i]["task_id"],
            "score": score,
            "failed_count": len(failed_tests)
        })

        if failed_tests:
            adversarial_bank.extend([{"task": dataset[i]["task_id"], "test": t} for t in failed_tests])

        print(f"Evaluated {dataset[i]['task_id']} | Score: {score:.1f}")

    # ==========================================
    # PRINTING THE CURRICULUM AND ADVERSARIAL BANK
    # ==========================================

    # Sort from highest score (Easiest) to lowest score (Hardest)
    curriculum_data.sort(key=lambda x: x["score"], reverse=True)
    seen_adversarial = set()
    unique_adversarial_bank = []
    for item in adversarial_bank:
        key = (item["task"], item["test"])
        if key not in seen_adversarial:
            seen_adversarial.add(key)
            unique_adversarial_bank.append(item)

    print("\n" + "=" * 60)
    print("📚 RL TRAINING CURRICULUM (EASY -> HARD)")
    print("=" * 60)
    for idx, item in enumerate(curriculum_data):
        difficulty = "🟢 EASY" if item["score"] > 80 else "🟡 MEDIUM" if item["score"] > 0 else "🔴 HARD"
        print(f"Step {idx + 1:<2} | {item['problem_id']:<20} | Score: {item['score']:>6.1f} | {difficulty}")

    print("\n" + "=" * 60)
    print("😈 ADVERSARIAL TEST BANK (Extracted Weaknesses)")
    print("=" * 60)
    # Print the top 5 unique adversarial edge cases
    for idx, adv in enumerate(unique_adversarial_bank[:5]):
        print(f"[{adv['task']}] Fails on: {adv['test'].strip()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
