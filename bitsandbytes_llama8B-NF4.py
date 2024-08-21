import os
from dotenv import load_dotenv
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


# Load the .env file
load_dotenv()

# Set environment variables
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['HF_HOME'] = '.'
hf_home = os.environ['HF_HOME'] ## Custom cache directory, **path to save the model**

# Set model and device
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_home = os.environ['HF_HOME']



# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    cache_dir=hf_home,
    device_map="auto",
    )

# Load model
## Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
    )
## Quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=hf_home,
    low_cpu_mem_usage=True,
    device_map="auto", ### Loads to multiple devices!!
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    )

# Save model
model.save_pretrained("models--fsaudm--Meta-Llama-3.1-8B-Instruct-NF4")
tokenizer.save_pretrained("models--fsaudm--Meta-Llama-3.1-8B-Instruct-NF4")


# Push model to hub
model.push_to_hub("fsaudm/Meta-Llama-3.1-8B-Instruct-NF4")
tokenizer.push_to_hub("fsaudm/Meta-Llama-3.1-8B-Instruct-NF4")


def generate_text(max_length, text):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate output
    generate_start = time.time()
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1
    )
    generate_end = time.time()
    
    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print(f"\nTime to generate output: {generate_end - generate_start:.2f} seconds")
    
    return 



# Generate text
generate_text(1000, "I want to be rich. How to? In 5 steps:")