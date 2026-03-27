import torch
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


# ==========================================
# 3. REAL GRADIENT ATTRIBUTION (POLICY GRADIENT)
# ==========================================
def extract_influential_tokens(model, tokenizer, prompt: str, generated_code: str, reward: float, device) -> list:
    """Uses a real PyTorch backward pass to calculate policy gradient attribution."""
    # Combine text and tokenize
    full_text = prompt + generated_code
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # Get embeddings and ensure we can track their gradients
    embeddings_layer = model.get_input_embeddings()
    input_embeds = embeddings_layer(inputs.input_ids).detach().clone()
    input_embeds.requires_grad_(True)

    # Forward pass using embeddings instead of input_ids so we can track the gradient
    outputs = model(inputs_embeds=input_embeds)

    # Calculate a simplified Policy Gradient Loss: -log(prob) * Advantage(reward)
    # We use the reward calculated from the test cases to scale the loss
    loss = -1.0 * outputs.logits.sum() * reward

    # Execute the backward pass to get the gradients
    model.zero_grad()
    loss.backward()

    # Calculate the L2 norm of the gradient for each token
    token_gradients = input_embeds.grad[0].norm(dim=-1)

    # We only care about the tokens the AI generated, not the prompt
    prompt_len = len(tokenizer(prompt).input_ids)
    gen_grads = token_gradients[prompt_len:]
    gen_tokens = inputs.input_ids[0][prompt_len:]

    # Extract the top 3 tokens with the highest gradient magnitude
    if len(gen_grads) > 0:
        top_indices = gen_grads.topk(min(3, len(gen_grads))).indices
        influential = [tokenizer.decode([gen_tokens[i]]) for i in top_indices]
        return [t.strip() for t in influential if t.strip()]  # Clean up whitespace
    return []


# ==========================================
# 4. FAILURE DASHBOARD (Expected vs Actual)
# ==========================================
def print_failure_dashboard(problem_id: str, code: str, failed_test: str, reward: float, bad_tokens: list):
    print("\n" + "!" * 60)
    print(f"🚨 FAILURE DASHBOARD | Problem: {problem_id}")
    print("!" * 60)
    print(f"💰 Reward Score:   {reward:.2f}")

    # Parse Expected vs Actual from the assert statement
    if "==" in failed_test:
        left_side = failed_test.split("==")[0].replace("assert", "").strip()
        expected = failed_test.split("==")[1].strip()
        print(f"❌ Failed Test:    {failed_test.strip()}")
        print(f"   ↳ Actual:       Evaluated {left_side}")
        print(f"   ↳ Expected:     {expected}")
    else:
        print(f"❌ Failed Test:    {failed_test.strip()}")

    print(f"🔍 Suspect Tokens: {bad_tokens} (Derived via Policy Gradient)")
    print("-" * 60)
    print("Code Snippet:")
    print("\n".join(code.strip().split('\n')[-4:]))
    print("!" * 60 + "\n")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print("Loading tokenizer and model for real PyTorch attribution mapping...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # We load the base model to run the backward pass
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Simulated prompt and hallucinated response
    prompt = "def multiply_list(numbers):\n"
    generated_code = """    result = 1
    for n in numbers:
        result = result + n  # BUG: Should be *
    return result
"""
    full_code = prompt + generated_code

    # An array of individual test cases
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