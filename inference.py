import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
# In a real scenario, this would point to your saved checkpoint folder
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 

def main():
    print("Loading model for reasoning evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Test Question (Math Reasoning)
    question = "If I have 3 apples and buy 4 more, but drop 1 on the way home, how many do I have?"
    
    # Format with <think> tag instructions
    messages = [
        {"role": "system", "content": "You are a math expert. Use <think> tags to show your reasoning."},
        {"role": "user", "content": question}
    ]

    # Generate Response
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    # Move to GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    model.to(device)

    print("\n--- Generating Response ---")
    outputs = model.generate(inputs, max_new_tokens=200, temperature=0.7)
    
    # Decode and Print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()
