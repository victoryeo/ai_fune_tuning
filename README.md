## This folder contains Supervised fine tuning files

- sff.py - run supervised fine tuning on existing Qwen model
- inference.py - run inference using the LORA adaptation output
- merge_model.py - generate a custom model on the LORA adaptation output
- inference_standalone.py - run inference using the merged model
- my_instruction_data.json - the custom data used to fine tune the existing model, used by sff.py
