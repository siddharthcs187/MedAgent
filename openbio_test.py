# import transformers
# import torch
# print(transformers.__version__)

# model_id = "aaditya/Llama3-OpenBioLLM-8B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cpu",
# )

# messages = [
#     {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
#     {"role": "user", "content": "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
# )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][len(prompt):])

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
