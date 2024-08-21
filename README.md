
# Meta-Llama-3.1-70B-Instruct-NF4 Text Generation

This repository contains a Python script and notebook that demonstrates how to load a 4-bit quantized Meta-Llama-3.1 model using the Hugging Face Transformers library for text generation tasks. The script loads the model, tokenizer, generates and streams text based on user input.

You can even run this notebook on Google Colab, making it simple to have your own Language Model (LLM) running for free!

### Quick example
```python
prompt = "Talk to me"
generate_response(prompt, model, tokenizer, max_length=100)
```

## Requirements

To get started in Google Colab, just run the first cell to install everything you need:

```python
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U bitsandbytes
```

## Usage
You'll need a Hugging Face token to download the models. You can get one [here](https://huggingface.co/docs/hub/en/security-tokens). Hugging Face is an amazing platform hosting millions of models, datasets, and tools for machine learning, so it's definitely worth checking out.

With quantization, we can make massive deep learning models more accessible to everyone. Now, you can easily run powerful models without needing super expensive hardware.
