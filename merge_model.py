import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv() # 👈 Load environment variables from .env file

base_model_id = "Qwen/Qwen2.5-7B"
adapter_dir = "./final_lora_adapter"
output_dir = "./my_custom_qwen_7b" # This will be your brand-new model folder

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
my_token = HUGGINGFACEHUB_API_TOKEN

print("1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=my_token)

print("2. Loading base model in full precision (BF16)...")
# Note: Ensure your GPU has enough VRAM (~15GB+) to load the unquantized model for merging.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="cpu",               # You can use "cpu" if your GPU VRAM is tight, it just takes longer
    torch_dtype=torch.bfloat16,
    token=my_token
)

print("3. Loading LoRA adapter onto base model...")
model = PeftModel.from_pretrained(base_model, adapter_dir)

print("4. Merging weights permanently...")
# This step mathematically collapses the LoRA matrices directly into the base weights
merged_model = model.merge_and_unload()

print(f"5. Saving your brand-new standalone model to {output_dir}...")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Done! You now have a custom, fully independent model.")