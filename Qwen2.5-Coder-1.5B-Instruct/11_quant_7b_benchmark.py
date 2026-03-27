import torch
import time
import traceback
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==========================================
# 1. EFFICIENT BATCHING
# ==========================================
def create_efficient_batches(dataset, tokenizer, batch_size):
    """Sorts problems by prompt length to minimize padding waste during batching."""
    lengths = []
    for item in dataset:
        # Measure token length
        tokenized_len = len(tokenizer(item["prompt"])["input_ids"])
        lengths.append((tokenized_len, item))

    # Sort from shortest to longest
    lengths.sort(key=lambda x: x[0])

    batches = []
    for i in range(0, len(lengths), batch_size):
        batches.append([x[1] for x in lengths[i:i + batch_size]])

    return batches


def estimate_trainable_footprint_gb(model) -> float:
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes / (1024 ** 3)


# ==========================================
# 2. INFERENCE PROFILING (BATCH SIZES)
# ==========================================
def profile_inference(model, tokenizer, dataset, batch_size, device):
    print(f"\n   --- Profiling Inference (Batch Size: {batch_size}) ---")
    batches = create_efficient_batches(dataset, tokenizer, batch_size)

    total_time = 0.0
    total_tokens = 0

    try:
        for batch in batches:
            prompts = [item["prompt"] for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            end_time = time.time()

            total_time += (end_time - start_time)
            input_len = inputs["input_ids"].shape[1]
            total_tokens += (outputs.shape[1] - input_len) * len(batch)

        print(
            f"   ✅ Throughput: {total_tokens / total_time:.2f} tokens/sec | Latency/Func: {total_time / len(dataset):.2f}s")
    except Exception as e:
        print(f"   ❌ INFERENCE CRASHED at Batch Size {batch_size}: {str(e).splitlines()[-1]}")


# ==========================================
# 3. TRAINING BENCHMARK (FP32 vs INT8)
# ==========================================
def benchmark_training(model_id, is_quantized, dataset, device):
    print(f"\n🚀 --- BENCHMARKING TRAINING: {model_id} (Quantized: {is_quantized}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if is_quantized:
            # The exact bitsandbytes int8 requirement
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # Full precision (fp32)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                local_files_only=True
            ).to(device)

        # Attach LoRA adapters so we can actually train the 7B model without instantly exploding memory
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                                 bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        estimated_size_gb = estimate_trainable_footprint_gb(model)
        print(f"   📦 Estimated Model Footprint: {estimated_size_gb:.2f} GB")
        if estimated_size_gb < 2.0:
            print("   ✅ Meets the '< 2GB' slim deployment target.")
        else:
            print("   ⚠️ Exceeds the '< 2GB' slim deployment target.")

        # Format the 50 HumanEval problems into a training dataset format (Prompt + Solution)
        def format_ds(example):
            text = example["prompt"] + example["canonical_solution"]
            return tokenizer(text, truncation=True, max_length=256, padding="max_length")

        train_data = dataset.map(format_ds, batched=True)

        # HuggingFace Trainer Setup
        training_args = TrainingArguments(
            output_dir="./benchmark_tmp",
            per_device_train_batch_size=1,
            num_train_epochs=1,  # Just 1 pass over the 50 problems
            max_steps=50,  # Force exactly 50 steps
            logging_steps=10,
            report_to="none"
        )

        trainer = Trainer(model=model, train_dataset=train_data, args=training_args)

        print("   ⏳ Starting training loop...")
        start_train = time.time()
        trainer.train()
        end_train = time.time()

        print(f"   ✅ Total Training Time (50 problems): {end_train - start_train:.2f} seconds")

        # Run Inference Profiling while the model is loaded
        for bs in [1, 4, 8]:
            profile_inference(model, tokenizer, dataset, bs, device)

    except Exception as e:
        print(f"\n🚨 HARDWARE/LIBRARY FAILURE TRIGGERED:")
        print(traceback.format_exc())


# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Host Device: {device}")

    dataset_split = os.getenv("TASK5_DATASET_SPLIT", "test[:50]")
    dataset = load_dataset("openai_humaneval", split=dataset_split)

    models_env = os.getenv("TASK5_MODELS")
    if models_env:
        models_to_test = [model_id.strip() for model_id in models_env.split(",") if model_id.strip()]
    else:
        models_to_test = [
            "codellama/CodeLlama-7b-hf",
            "bigcode/starcoderbase-7b"
        ]

    for model_id in models_to_test:
        print("\n" + "=" * 70)
        print(f"EVALUATING MODEL: {model_id}")
        print("=" * 70)

        # Test 1: Full Precision (fp32)
        benchmark_training(model_id, is_quantized=False, dataset=dataset, device=device)

        # Test 2: Quantized (int8 bitsandbytes)
        benchmark_training(model_id, is_quantized=True, dataset=dataset, device=device)


if __name__ == "__main__":
    main()
