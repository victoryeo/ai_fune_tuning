import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Point directly to your local merged model folder
model_path = "./my_custom_qwen_7b"

print("Loading tokenizer from custom model folder...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Configuring 4-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print("Loading your brand-new custom model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=quantization_config
)

# 2. Prepare your test prompt (Keep the exact template used in training)
test_instruction = "What is the company policy on annual leave?"
prompt = f"### Instruction:\n{test_instruction}\n\n### Response:\n"

print("\nGenerating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Switch model to evaluation mode
model.eval()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

# 3. Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*40)
print("CUSTOM STANDALONE MODEL OUTPUT:")
print("="*40)
print(generated_text)
print("="*40)