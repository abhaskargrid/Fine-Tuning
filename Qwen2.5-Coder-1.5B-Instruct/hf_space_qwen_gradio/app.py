import os
import subprocess
import tempfile
from functools import lru_cache

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = APP_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def execute_code(code: str, test_cases: str = "") -> tuple[bool, str]:
    full_code = f"{code}\n\n{test_cases}".strip() + "\n"
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(full_code.encode("utf-8"))
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            timeout=5,
            text=True,
        )
        combined_output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, combined_output
    except Exception as exc:
        return False, str(exc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class MCTSNode:
    def __init__(self, code: str = "", parent=None):
        self.code = code
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0


@lru_cache(maxsize=1)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(ADAPTER_DIR)


@lru_cache(maxsize=1)
def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=MODEL_DTYPE,
    ).to(DEVICE)


@lru_cache(maxsize=1)
def load_tuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=MODEL_DTYPE,
    )
    return PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)


def mode_visibility(model_mode: str):
    show_base = model_mode in {"Base", "Both"}
    show_tuned = model_mode in {"Tuned", "Both"}
    return (
        gr.update(visible=show_base),
        gr.update(visible=show_tuned),
    )


def run_search(
    model,
    tokenizer,
    prompt,
    tests,
    num_simulations,
    max_tokens,
    temperature,
    status_prefix,
    progress=gr.Progress(track_tqdm=False),
):
    root = MCTSNode(code=prompt)
    best_node = None
    best_result = ""
    start_time = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if DEVICE == "cuda" else None
    wall_start = None

    if DEVICE == "cuda":
        start_time.record()
    else:
        import time
        wall_start = time.perf_counter()

    for step in range(num_simulations):
        progress((step, num_simulations), desc=f"{status_prefix}: candidate {step + 1}/{num_simulations}")
        current = root
        inputs = tokenizer(current.code, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
            generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

        success, test_output = execute_code(generated_code, tests)
        reward = 1.0 if success else 0.0

        new_node = MCTSNode(code=generated_code, parent=current)
        new_node.reward = reward
        new_node.visits = 1
        current.children.append(new_node)

        while current:
            current.visits += 1
            current.reward += reward
            current = current.parent

        best_result = test_output or best_result
        if success:
            best_node = new_node
            best_result = test_output or "All tests passed."
            break

    if best_node is None:
        best_node = max(root.children, key=lambda node: node.reward / (node.visits + 1e-6)) if root.children else root

    if DEVICE == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed_seconds = start_time.elapsed_time(end_time) / 1000.0
    else:
        import time
        elapsed_seconds = time.perf_counter() - wall_start

    progress((num_simulations, num_simulations), desc=f"{status_prefix}: done")
    return best_node.code, best_result or "No passing candidate found.", elapsed_seconds


def generate_code(prompt, tests, model_mode, num_simulations, temperature, max_tokens, progress=gr.Progress(track_tqdm=False)):
    tokenizer = load_tokenizer()

    outputs = {
        "Base Model": "",
        "Tuned Model": "",
    }
    summary_lines = []

    base_visible, tuned_visible = mode_visibility(model_mode)
    yield (
        base_visible,
        tuned_visible,
        outputs["Base Model"],
        outputs["Tuned Model"],
        "",
    )

    run_order = []
    if model_mode == "Tuned":
        run_order = [("Tuned Model", load_tuned_model)]
    elif model_mode == "Base":
        run_order = [("Base Model", load_base_model)]
    else:
        run_order = [("Tuned Model", load_tuned_model), ("Base Model", load_base_model)]

    for label, loader in run_order:
        model = loader()
        code, verification, elapsed_seconds = run_search(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            tests=tests,
            num_simulations=int(num_simulations),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            status_prefix=label,
            progress=progress,
        )

        outputs[label] = code
        summary_lines.append(f"{label}: {elapsed_seconds:.2f}s")
        summary_lines.append(verification)

        yield (
            base_visible,
            tuned_visible,
            outputs["Base Model"],
            outputs["Tuned Model"],
            "\n\n".join(summary_lines),
        )


DEFAULT_PROMPT = """from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\""""

DEFAULT_TESTS = """def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True

check(has_close_elements)"""


with gr.Blocks(title="Qwen2.5-Coder LoRA Playground") as demo:
    gr.Markdown(
        """
        # Qwen2.5-Coder LoRA Playground
        Compare the base `Qwen/Qwen2.5-Coder-1.5B-Instruct` model with the fine-tuned LoRA adapter
        using a lightweight search loop inspired by the original Streamlit prototype.
        """
    )

    with gr.Row():
        model_mode = gr.Radio(
            choices=["Tuned", "Base", "Both"],
            value="Tuned",
            label="Model Mode",
        )
        num_simulations = gr.Slider(1, 20, value=7, step=1, label="Max Search Branches")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Code(value=DEFAULT_PROMPT, language="python", label="Function Prompt")
            tests = gr.Code(value=DEFAULT_TESTS, language="python", label="Unit Tests")
            with gr.Row():
                run_btn = gr.Button("Generate & Verify Code", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
        with gr.Column(scale=1):
            with gr.Group(visible=False) as base_group:
                gr.Markdown("### Base Model Output")
                base_output = gr.Code(language="python", label="Base Model")
            with gr.Group(visible=True) as tuned_group:
                gr.Markdown("### Tuned Model Output")
                tuned_output = gr.Code(language="python", label="Tuned Model")
            verification_output = gr.Textbox(label="Run Summary", lines=10)

    model_mode.change(
        fn=mode_visibility,
        inputs=model_mode,
        outputs=[base_group, tuned_group],
    )

    generation_event = run_btn.click(
        fn=generate_code,
        inputs=[prompt, tests, model_mode, num_simulations, temperature, max_tokens],
        outputs=[base_group, tuned_group, base_output, tuned_output, verification_output],
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[generation_event])


if __name__ == "__main__":
    demo.launch(
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
