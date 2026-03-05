import streamlit as st
import torch
import math
import random
import subprocess
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR = "./lora-finetuned"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    st.info("Loading model to MPS... this takes a moment.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)
    return model, tokenizer


def execute_code(code, test_cases=""):
    """Runs code in a temporary sandbox and returns pass/fail."""
    full_code = f"{code}\n\n{test_cases}"
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(full_code.encode())
        tmp_path = tmp.name

    try:
        result = subprocess.run(["python3", tmp_path], capture_output=True, timeout=5, text=True)
        os.remove(tmp_path)
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)


# --- SIMPLE MCTS LOGIC ---
class MCTSNode:
    def __init__(self, code="", parent=None):
        self.code = code
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def uct_score(self, total_visits):
        if self.visits == 0: return float('inf')
        return (self.reward / self.visits) + 1.41 * math.sqrt(math.log(total_visits) / self.visits)


# --- STREAMLIT UI ---
st.set_page_config(page_title="Qwen MCTS Code Gen", layout="wide")

st.title("🚀 Qwen2.5-Coder + MCTS Code Search")
st.markdown(f"Running on: **{DEVICE.upper()}** | Model: `{MODEL_ID}` (with LoRA)")

# Move settings to the main page in 3 columns
st.markdown("### ⚙️ Search Settings")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    num_simulations = st.slider("Max Search Branches (MCTS)", min_value=1, max_value=20, value=10)
with col_s2:
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
with col_s3:
    max_tokens = st.slider("Max Tokens", min_value=64, max_value=1024, value=256, step=64)

st.markdown("---")

col1, col2 = st.columns([1, 1])

# Pre-filled HumanEval/0 Data
default_prompt = '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """'''

default_tests = '''def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

# Call the check function with the main function
check(has_close_elements)'''

with col1:
    prompt = st.text_area("Function Prompt:", default_prompt, height=250)
    tests = st.text_area("Unit Tests (Asserts):", default_tests, height=250)
    run_btn = st.button("Generate & Verify Code", type="primary", use_container_width=True)

if run_btn:
    model, tokenizer = load_model()
    root = MCTSNode(code=prompt)

    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_node = None
    success_found = False

    for i in range(num_simulations):
        status_text.text(f"Simulation {i + 1}/{num_simulations}: Generating branch...")

        # 1. Selection & Expansion
        current = root
        inputs = tokenizer(current.code, return_tensors="pt").to(DEVICE)

        # 2. Simulation (Rollout) - Using the UI parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.95
            )
            # Decode only the newly generated tokens
            gen_code = tokenizer.decode(output[0], skip_special_tokens=True)

        # 3. Evaluation (Reward)
        status_text.text(f"Simulation {i + 1}/{num_simulations}: Running tests on generated code...")
        success, error_log = execute_code(gen_code, tests)
        reward = 1.0 if success else 0.0

        # 4. Backpropagation
        new_node = MCTSNode(code=gen_code, parent=current)
        new_node.reward = reward
        new_node.visits = 1
        current.children.append(new_node)

        # Update tree metrics
        while current:
            current.visits += 1
            current.reward += reward
            current = current.parent

        progress_bar.progress((i + 1) / num_simulations)

        if success:
            st.success(f"🎯 Valid solution found on simulation branch {i + 1}!")
            best_node = new_node
            success_found = True
            break  # Exit early if we found a perfect solution!

    status_text.text("Search Complete.")

    # Show Results
    if not best_node:
        if root.children:
            best_node = max(root.children, key=lambda x: x.reward / (x.visits + 1e-6))
        else:
            best_node = root

    with col2:
        st.subheader("Best Generated Output")
        st.code(best_node.code, language="python")

        final_success, final_err = execute_code(best_node.code, tests)
        if final_success:
            st.balloons()
            st.success("✅ Unit Tests Passed!")
        else:
            st.error("❌ Unit Tests Failed.")
            with st.expander("View Error Log"):
                st.code(final_err, language="bash")