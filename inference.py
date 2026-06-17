import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Added BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv() # 👈 Load environment variables from .env file

# 1. Setup paths and token
adapter_dir = "./final_lora_adapter" 
base_model_id = "Qwen/Qwen2.5-7B"

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
my_token = HUGGINGFACEHUB_API_TOKEN

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=my_token)

print("Configuring 4-bit quantization...")
# Recreate the 4-bit configuration to match training setup exactly
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    dtype=torch.bfloat16, # Fixed: Changed torch_dtype -> dtype
    quantization_config=quantization_config # Fixed: Replaced load_in_4bit=True
)

print("Merging LoRA adapter onto base model...")
model = PeftModel.from_pretrained(model, adapter_dir)

# 2. Prepare your test prompt
# CRITICAL: This layout must match the template you used in formatting_prompts_func!
test_instruction = "What is the company policy on annual leave?" 

prompt = f"### Instruction:\n{test_instruction}\n\n### Response:\n"

# 3. Tokenize and generate response
print("\nGenerating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Switch model to evaluation mode
model.eval()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,       # Maximum length of the generated answer
        temperature=0.7,          # Controls creativity (lower is more deterministic)
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

# 4. Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*40)
print("MODEL OUTPUT:")
print("="*40)
print(generated_text)
print("="*40)