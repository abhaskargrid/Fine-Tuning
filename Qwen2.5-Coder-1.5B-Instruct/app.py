import streamlit as st
import torch
import math
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
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@st.cache_resource
def load_base_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)


@st.cache_resource
def load_tuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    return PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)


def execute_code(code, test_cases=""):
    """Runs code in a temporary sandbox and returns pass/fail."""
    full_code = f"{code}\n\n{test_cases}"
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(full_code.encode())
        tmp_path = tmp.name
    try:
        result = subprocess.run(["python3", tmp_path], capture_output=True, timeout=5, text=True)
        os.remove(tmp_path)
        return result.returncode == 0
    except Exception:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return False


# --- SIMPLE MCTS LOGIC ---
class MCTSNode:
    def __init__(self, code="", parent=None):
        self.code = code
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0


def run_mcts_generation(model, tokenizer, prompt, tests, num_simulations, max_tokens, temperature):
    root = MCTSNode(code=prompt)
    best_node = None

    for _ in range(num_simulations):
        current = root
        inputs = tokenizer(current.code, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.95
            )
            gen_code = tokenizer.decode(output[0], skip_special_tokens=True)

        success = execute_code(gen_code, tests)
        reward = 1.0 if success else 0.0

        new_node = MCTSNode(code=gen_code, parent=current)
        new_node.reward = reward
        new_node.visits = 1
        current.children.append(new_node)

        while current:
            current.visits += 1
            current.reward += reward
            current = current.parent

        if success:
            best_node = new_node
            break

    if not best_node:
        best_node = max(root.children, key=lambda x: x.reward / (x.visits + 1e-6)) if root.children else root

    return best_node.code


# --- STREAMLIT UI ---
st.set_page_config(page_title="Qwen MCTS Code Gen", layout="wide")

st.title("🚀 Qwen2.5-Coder + MCTS Code Search")

# Settings in 3 columns
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    num_simulations = st.slider("Max Search Branches (MCTS)", min_value=1, max_value=20, value=10)
with col_s2:
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
with col_s3:
    max_tokens = st.slider("Max Tokens", min_value=64, max_value=1024, value=256, step=64)
model_mode = st.radio("Model Mode", ["Tuned", "Base", "Both"], horizontal=True)

st.markdown("---")
col1, col2 = st.columns([1, 1])

# --- STARTING CODE DEFAULTS ---
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

check(has_close_elements)'''

with col1:
    prompt = st.text_area("Function Prompt:", value=default_prompt, height=250)
    tests = st.text_area("Unit Tests (Asserts):", value=default_tests, height=250)
    run_btn = st.button("Generate & Verify Code", type="primary", use_container_width=True)

if run_btn:
    tokenizer = load_tokenizer()
    mode_to_runs = {
        "Tuned": [("Tuned Model", load_tuned_model)],
        "Base": [("Base Model", load_base_model)],
        "Both": [("Base Model", load_base_model), ("Tuned Model", load_tuned_model)]
    }
    selected_runs = mode_to_runs[model_mode]

    progress_bar = st.progress(0)
    status_text = st.empty()
    generated_outputs = []
    total_steps = len(selected_runs)

    for idx, (label, model_loader) in enumerate(selected_runs):
        status_text.text(f"Running {label} ({idx + 1}/{total_steps})...")
        model = model_loader()
        code = run_mcts_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            tests=tests,
            num_simulations=num_simulations,
            max_tokens=max_tokens,
            temperature=temperature
        )
        generated_outputs.append((label, code))
        progress_bar.progress((idx + 1) / total_steps)

    status_text.empty()
    progress_bar.empty()

    with col2:
        if len(generated_outputs) == 1:
            st.subheader(f"Generated Code ({generated_outputs[0][0]})")
            st.code(generated_outputs[0][1], language="python")
        else:
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.subheader(generated_outputs[0][0])
                st.code(generated_outputs[0][1], language="python")
            with out_col2:
                st.subheader(generated_outputs[1][0])
                st.code(generated_outputs[1][1], language="python")
