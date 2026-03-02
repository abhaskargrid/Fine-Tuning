import streamlit as st
import torch
import subprocess
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================================
# 1. LOAD MODEL (Cached so it only loads once)
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    adapter_dir = "./qwen-lora-finetuned"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

    return tokenizer, model, device


tokenizer, model, device = load_model()


# ==========================================
# 2. SAFE TEST RUNNER
# ==========================================
def run_tests_safely(generated_code, test_code):
    """Saves code to a temp file and runs it to prevent UI crashes."""
    full_script = generated_code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_script)
        temp_path = f.name

    try:
        # Run the code with a 5-second timeout (prevents infinite loops)
        result = subprocess.run(["python3", temp_path], capture_output=True, text=True, timeout=5)
        os.remove(temp_path)

        if result.returncode == 0:
            return True, "All tests passed! ✅\n" + result.stdout
        else:
            return False, "Tests failed. ❌\n" + result.stderr
    except subprocess.TimeoutExpired:
        os.remove(temp_path)
        return False, "Execution timed out (Infinite loop detected). ❌"


# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="My Fine-Tuned AI Coder", layout="wide")
st.title("🚀 Custom Qwen LoRA Coder + Search")
st.markdown("Watch the fine-tuned model generate code and verify it against unit tests in real-time.")

# Sidebar Settings
st.sidebar.header("Search Settings")
search_iterations = st.sidebar.slider("Max Search Branches (MCTS depth)", 1, 10, 5)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.1, 1.0, 0.4)
max_tokens = st.sidebar.slider("Max Tokens", 128, 512, 256)

# Main UI Inputs
col1, col2 = st.columns(2)
with col1:
    prompt = st.text_area("Python Prompt / Docstring", height=250, value='''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. """
''')

with col2:
    unit_tests = st.text_area("Hidden Unit Tests (Verification)", height=250, value='''
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
print("Tests passed!")
''')

# ==========================================
# 4. EXECUTION & MCTS LOGIC
# ==========================================
if st.button("Generate & Search", type="primary"):

    # UI Elements for real-time updates
    status_text = st.empty()
    progress_bar = st.progress(0)
    log_area = st.expander("Search Logs", expanded=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    success = False
    best_code = ""

    # The Search Loop
    for iteration in range(search_iterations):
        status_text.markdown(f"**Searching branch {iteration + 1}/{search_iterations}...**")
        progress_bar.progress((iteration) / search_iterations)

        # 1. Generate a path
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=0.95
        )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_code = prompt + completion

        # 2. Evaluate the path (Test Runner)
        passed, test_output = run_tests_safely(generated_code, unit_tests)

        # 3. Log the branch results
        with log_area:
            st.markdown(f"**Attempt {iteration + 1}:**")
            if passed:
                st.success(test_output)
            else:
                st.error(test_output.split("\n")[
                             -2] if "\n" in test_output else test_output)  # Just show the last error line

        # 4. Search halting condition
        if passed:
            success = True
            best_code = generated_code
            progress_bar.progress(1.0)
            status_text.markdown("**Search Complete! Valid path found. 🎉**")
            break

        # Keep the last code just in case it never passes
        best_code = generated_code

    if not success:
        progress_bar.progress(1.0)
        status_text.markdown("**Search Exhausted. No fully valid path found. ⚠️**")

    # Display the final code
    st.subheader("Final Output Code:")
    st.code(best_code, language="python")