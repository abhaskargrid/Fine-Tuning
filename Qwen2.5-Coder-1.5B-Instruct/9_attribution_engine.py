import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. BINARY SEARCH TEST ISOLATOR
# ==========================================
def run_test_block(code_str: str, test_cases: list) -> bool:
    """Helper to run a block of tests and return True if they all pass."""
    combined_tests = "\n".join(test_cases)
    full_code = f"{code_str}\n{combined_tests}"
    try:
        # Using exec for local speed in our attribution engine
        exec(full_code, {})
        return True
    except AssertionError:
        return False
    except Exception:
        return False  # Syntax or runtime crash


def find_failing_test_binary_search(code_str: str, test_cases: list) -> str:
    """Uses O(log N) binary search to isolate the exact failing assert statement."""
    if not test_cases:
        return None
    if len(test_cases) == 1:
        return test_cases[0] if not run_test_block(code_str, test_cases) else None

    mid = len(test_cases) // 2
    left_half = test_cases[:mid]
    right_half = test_cases[mid:]

    # Check left half
    if not run_test_block(code_str, left_half):
        return find_failing_test_binary_search(code_str, left_half)
    # If left passes, the bug MUST be in the right half
    elif not run_test_block(code_str, right_half):
        return find_failing_test_binary_search(code_str, right_half)

    return None  # All passed!


# ==========================================
# 2. GRANULAR REWARD CALCULATOR
# ==========================================
def calculate_granular_reward(code_str: str, test_cases: list) -> float:
    """Calculates reward: +1 for pass, -0.25 per failed assertion."""
    reward = 0.0
    for test in test_cases:
        if run_test_block(code_str, [test]):
            reward += 1.0
        else:
            reward -= 0.25
    return reward


def evaluate_expression(code_str: str, expression: str):
    namespace = {}
    exec(code_str, namespace)
    return eval(expression, namespace)


# ==========================================
# 3. REAL GRADIENT ATTRIBUTION (POLICY GRADIENT)
# ==========================================
def extract_influential_tokens(model, tokenizer, prompt: str, generated_code: str, reward: float, device) -> list:
    """Use generated-token log-probs as a simple policy-gradient style attribution signal."""
    full_text = prompt + generated_code
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    embeddings_layer = model.get_input_embeddings()
    input_embeds = embeddings_layer(inputs.input_ids).detach().clone()
    input_embeds.requires_grad_(True)

    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.logits[:, :-1, :]
    target_ids = inputs.input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = len(tokenizer(prompt).input_ids)
    gen_start = max(prompt_len - 1, 0)
    gen_token_log_probs = token_log_probs[0, gen_start:]

    if gen_token_log_probs.numel() == 0:
        return []

    advantage = reward if reward != 0 else -0.25
    loss = -(gen_token_log_probs.sum() * advantage)

    model.zero_grad()
    loss.backward()

    token_gradients = input_embeds.grad[0].norm(dim=-1)
    gen_grads = token_gradients[prompt_len:]
    gen_tokens = inputs.input_ids[0][prompt_len:]

    if len(gen_grads) > 0:
        top_indices = gen_grads.topk(min(3, len(gen_grads))).indices
        influential = []
        for idx in top_indices.tolist():
            token_text = tokenizer.decode([gen_tokens[idx]]).strip()
            influential.append({
                "token": token_text or repr(tokenizer.decode([gen_tokens[idx]])),
                "score": float(gen_grads[idx].item())
            })
        return influential
    return []


# ==========================================
# 4. FAILURE DASHBOARD (Expected vs Actual)
# ==========================================
def print_failure_dashboard(problem_id: str, code: str, failed_test: str, reward: float, bad_tokens: list):
    print("\n" + "!" * 60)
    print(f"🚨 FAILURE DASHBOARD | Problem: {problem_id}")
    print("!" * 60)
    print(f"💰 Reward Score:   {reward:.2f}")

    if "==" in failed_test:
        left_side = failed_test.split("==")[0].replace("assert", "").strip()
        expected = failed_test.split("==")[1].strip()
        try:
            actual = evaluate_expression(code, left_side)
        except Exception as exc:
            actual = f"<evaluation failed: {exc}>"
        print(f"❌ Failed Test:    {failed_test.strip()}")
        print(f"   ↳ Actual:       {actual}")
        print(f"   ↳ Expected:     {expected}")
    else:
        print(f"❌ Failed Test:    {failed_test.strip()}")

    print(f"🔍 Suspect Tokens: {bad_tokens} (Derived via token log-prob gradients)")
    print("-" * 60)
    print("Code Snippet:")
    print("\n".join(code.strip().split('\n')[-4:]))
    print("!" * 60 + "\n")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print("Loading tokenizer and model for real PyTorch attribution mapping...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # We load the base model to run the backward pass
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).to(device)

    prompt = os.getenv("TASK2_PROMPT", "def multiply_list(numbers):\n")
    generated_code = os.getenv(
        "TASK2_GENERATED_CODE",
        """    result = 1
    for n in numbers:
        result = result + n  # BUG: Should be *
    return result
"""
    )
    full_code = prompt + generated_code

    tests = [
        "assert multiply_list([1, 1, 1]) == 1",
        "assert multiply_list([2, 2]) == 4",
        "assert multiply_list([0, 5]) == 0",
    ]

    # 1. Calculate Reward
    reward = calculate_granular_reward(full_code, tests)

    # 2. Binary Search to isolate the first failure
    failed_test = find_failing_test_binary_search(full_code, tests)

    # 3. Extract Token Gradients (Attribution)
    bad_tokens = []
    if failed_test:
        bad_tokens = extract_influential_tokens(model, tokenizer, prompt, generated_code, reward, device)

    # 4. Render Dashboard
    if failed_test:
        print_failure_dashboard("multiply_list", full_code, failed_test, reward, bad_tokens)
    else:
        print("✅ All tests passed!")


if __name__ == "__main__":
    main()
