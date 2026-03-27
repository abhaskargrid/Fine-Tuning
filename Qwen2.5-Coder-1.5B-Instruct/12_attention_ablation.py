import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. THE ABLATION ENGINE
# ==========================================
def remove_docstring(prompt: str) -> str:
    no_docstring = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', prompt)
    no_docstring = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', no_docstring)
    return os.linesep.join([s for s in no_docstring.splitlines() if s.strip()])


def remove_type_hints(prompt: str) -> str:
    cleaned = re.sub(r'->\s*[^:]+:', ':', prompt)
    cleaned = re.sub(r':\s*[A-Za-z_][A-Za-z0-9_\[\]]*\s*(?=[,\)])', '', cleaned)
    return cleaned


def run_in_sandbox(generated_code: str, test_code: str) -> bool:
    full_code = generated_code + "\n\n" + test_code
    try:
        result = subprocess.run(
            ["python3", "4_sandbox.py"],
            input=full_code,
            text=True,
            capture_output=True,
            timeout=7
        )
        return result.returncode == 0 and "SUCCESS" in result.stdout
    except Exception:
        return False


# ==========================================
# 2. THE ATTENTION VISUALIZER
# ==========================================
def plot_attention_heatmap(model, tokenizer, prompt, problem_id, outcome_label, device):
    print(f"🔍 Extracting Neural Attention Weights for {problem_id} ({outcome_label})...")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get the LAST layer, FIRST batch item
    last_layer_attn = outputs.attentions[-1][0]

    # Average across all attention heads and ensure it's standard float32 on CPU
    avg_attention = last_layer_attn.mean(dim=0).to(torch.float32).cpu().numpy()

    # Slice the last 30 tokens
    N = min(30, avg_attention.shape[0])
    focus_attn = avg_attention[-N:, -N:]

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[-N:]
    clean_tokens = [t.replace("Ġ", " ") for t in tokens]
    clean_tokens = [t.replace("\n", "↵") for t in clean_tokens]

    # Plot the Data
    plt.figure(figsize=(10, 8))
    # We remove the hardcoded cmap constraints so seaborn scales to the actual data!
    sns.heatmap(focus_attn, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="magma")

    plt.title(f"Model Attention Heatmap - {problem_id} ({outcome_label})", fontsize=14, fontweight='bold')
    plt.xlabel("Key Token (What the AI is looking at)", fontsize=12)
    plt.ylabel("Query Token (The current word being processed)", fontsize=12)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = f"attention_heatmap_{problem_id.replace('/', '_')}_{outcome_label.lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"   ✅ Successfully saved heatmap to: {filename}")


# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # THE FIX: Force CPU and float32 to bypass the Mac MPS NaN bug
    device = torch.device("cpu")
    print(f"Host Device: {device} (Forced for numeric stability)")

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    print("Loading Model with Neural Intercepts Enabled (Eager + FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # THE FIX
        attn_implementation="eager",
        local_files_only=True
    ).to(device)

    dataset_split = os.getenv("TASK4_DATASET_SPLIT", "test[:3]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)

    print("\n" + "=" * 60)
    print("🧠 RUNNING ATTENTION EXTRACTION")
    print("=" * 60)

    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        problem_id = dataset[i]["task_id"]
        entry_point = dataset[i]["entry_point"]
        test_code = dataset[i]["test"] + f"\ncheck({entry_point})"

        print(f"\n--- {problem_id} ---")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.2
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_code = prompt + completion
        outcome_label = "success" if run_in_sandbox(generated_code, test_code) else "failure"

        plot_attention_heatmap(model, tokenizer, prompt, problem_id, outcome_label, device)


if __name__ == "__main__":
    main()
