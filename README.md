
# Meta-Llama-3.1-8B-Instruct-NF4

This repository showcases how to load and use a 4-bit quantized Meta-Llama-3.1-Instruct 8B model for text generation with Hugging Face Transformers. Alongside the notebook, the repository also includes a script that walks you through the process of quantizing the model using the powerful QLoRA algorithm. Both the inference and quantization steps are designed to be accessible, even on free-tier Google Colab GPUs.

The NF4 quantized model uses just under 6 GB of VRAM, making it feasible to load and run inference on free-tier Google Colab GPUs. This quantization technique significantly reduces the resource requirements while maintaining the model's performance

**Model card:**
[fsaudm/Meta-Llama-3.1-8B-Instruct-NF4](fsaudm/Meta-Llama-3.1-8B-Instruct-NF4)

Loading times on Colab:
- tokenizer: 3.87 seconds
- model: 221.83 seconds (download included)


## Requirements
To get started with text generation and quantization, this will install everything you need:

```python
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U bitsandbytes
```

## Usage
You'll need a Hugging Face token to download the models. You can get one [here](https://huggingface.co/docs/hub/en/security-tokens). Hugging Face is an amazing platform hosting millions of models, datasets, and tools for machine learning, so it's definitely worth checking out.


### Quick example
```python
prompt = "Talk to me"
generate_response(prompt, model, tokenizer, max_length=100)
```

## Quantization
The Llama-3.1 8B model was quantized to 4-bit precision using the QLoRA algorithm and the bitsandbytes implementation.  This script allows you to replicate the quantization on this and other models following the same 4-bit configuration. The quantization_config used:

```
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
    )
```

You can explore this and other models [here](https://huggingface.co/fsaudm).



