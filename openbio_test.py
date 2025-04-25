#!pip install transformers accelerate -q

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure you're on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_id = "aaditya/Llama3-OpenBioLLM-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model in float16 only if using GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Create prompt manually (since tokenizer.chat_template may not exist)
prompt = """<|system|>
You are a helpful medical assistant developed by Saama AI Labs.
<|user|>
How can I split a 3mg or 4mg Waefin pill so I can get a 2.5mg dose?
<|assistant|>
"""

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("aaditya/Llama3-OpenBioLLM-8B", trust_remote_code=True)

# prompt = "I have a cold what do I do ?"
# inputs = tokenizer(prompt, return_tensors="pt")
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure you're on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_id = "aaditya/Llama3-OpenBioLLM-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model in float16 only if using GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Create prompt manually (since tokenizer.chat_template may not exist)
prompt = """<|system|>
You are a helpful medical assistant developed by Saama AI Labs.
<|user|>
How can I split a 3mg or 4mg Waefin pill so I can get a 2.5mg dose?
<|assistant|>
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode and print
response = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print("Assistant:", response)

# Decode and print
response = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print("Assistant:", response)