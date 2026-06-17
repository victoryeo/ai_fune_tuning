import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv() # 👈 Load environment variables from .env file

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

my_token = HUGGINGFACEHUB_API_TOKEN
print("Hugging Face API Token:", my_token)

# 1. Load Model and Tokenizer
print("Load Model and Tokenizer")
#model_id = "meta-llama/Meta-Llama-3-8B"
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=my_token)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit to save VRAM
print("Load model in 4-bit to save VRAM")
# Define the explicit 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",             # Recommended for LLMs
    bnb_4bit_use_double_quant=True         # Saves extra VRAM
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=quantization_config
)

# 2. Configure LoRA
print("Configure LORA")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Load Dataset
dataset = load_dataset("json", data_files="my_instruction_data.json", split="train")

# Define how to format data rows into the prompt template
# 3. Load Dataset
dataset = load_dataset("json", data_files="my_instruction_data.json", split="train")

# Define how to format a SINGLE data row into the prompt template
def formatting_prompts_func(example):
    # Process a single row directly (no loops needed!)
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# 4. Set Training Hyperparameters
training_args = SFTConfig(
    output_dir="./instruction_tuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    report_to="none",
    max_length=512
)

# 5. Initialize SFTTrainer and Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()

# 6. Save the trained adapters
trainer.model.save_pretrained("./final_lora_adapter")
