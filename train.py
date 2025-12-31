import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # Using 0.5B for fast demo training
OUTPUT_DIR = "qwen-grpo-math"

def main():
    # 1. Load Tokenizer & Model
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. Load Dataset (GSM8K for Math Reasoning)
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train").select(range(200)) # Small subset for demo

    def format_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": "You are a math expert. Solve this problem step-by-step wrapped in <think> tags."},
                {"role": "user", "content": example["question"]}
            ],
            "answer": example["answer"]
        }
    
    dataset = dataset.map(format_prompt)

    # 3. Define Reward Functions (RLHF Core)
    def format_reward(completions, **kwargs):
        """Reward model for using correct <think>...</think> structure."""
        rewards = []
        for text in completions:
            if "<think>" in text and "</think>" in text:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def accuracy_reward(completions, answer, **kwargs):
        """Reward model for getting the correct numeric answer."""
        rewards = []
        for text, correct_ans in zip(completions, answer):
            # Extract the number after "####" (GSM8K standard format)
            correct_num = correct_ans.split("####")[-1].strip()
            if correct_num in text:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # 4. LoRA Configuration (Memory Efficiency)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments (GRPO)
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=50, # Short run for demo
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
    )

    # 6. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # 7. Start Training
    print("Starting GRPO Fine-Tuning...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Training Complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
